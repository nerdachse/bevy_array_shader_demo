use bevy::asset::{LoadState, LoadedFolder};
use bevy::pbr::light_consts::lux;
use bevy::pbr::wireframe::{Wireframe, WireframePlugin};
use bevy::prelude::*;
use bevy::render::mesh::{Indices, MeshVertexAttribute, VertexAttributeValues};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, PrimitiveTopology, ShaderRef, TextureDimension, VertexFormat,
};
use bevy::render::texture::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor};
use bevy_flycam::PlayerPlugin;
use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use block_mesh::{
    greedy_quads, GreedyQuadsBuffer, MergeVoxel, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};

#[derive(Default, States, Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum AppState {
    #[default]
    Loading,
    PrepareTextureArray,
    Run,
}

pub const ATTRIBUTE_DATA: MeshVertexAttribute =
    MeshVertexAttribute::new("Vertex_Data", 0x696969, VertexFormat::Uint32);

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "greedy meshing texturing experiment".into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
        )
        .add_plugins(WireframePlugin)
        .add_plugins(MaterialPlugin::<ArrayTextureMaterial>::default())
        .add_plugins(PlayerPlugin)
        .init_state::<AppState>()
        .add_systems(OnEnter(AppState::Loading), start_load_images)
        .add_systems(
            OnEnter(AppState::PrepareTextureArray),
            process_loaded_images_as_array_texture,
        )
        .add_systems(
            Update,
            check_all_images_loaded.run_if(in_state(AppState::Loading)),
        )
        .add_systems(OnEnter(AppState::Run), setup)
        .run();
}

#[derive(Resource)]
struct BlockImageFolder(Handle<LoadedFolder>);

fn start_load_images(mut cmd: Commands, ass: Res<AssetServer>) {
    cmd.insert_resource(BlockImageFolder(ass.load_folder("block-images")));
}

fn check_all_images_loaded(
    mut next_state: ResMut<NextState<AppState>>,
    mut events: EventReader<AssetEvent<LoadedFolder>>,
    block_image_folder: ResMut<BlockImageFolder>,
) {
    for event in events.read() {
        if event.is_loaded_with_dependencies(&block_image_folder.0) {
            next_state.set(AppState::PrepareTextureArray);
        }
    }
}

fn process_loaded_images_as_array_texture(
    mut next_state: ResMut<NextState<AppState>>,
    mut cmd: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<ArrayTextureMaterial>>,
    loaded_folders: Res<Assets<LoadedFolder>>,
    block_image_folder: Res<BlockImageFolder>,
) {
    let mut loaded_images: Vec<&Image> = Vec::new();
    let loaded_folder: &LoadedFolder = loaded_folders
        .get(&block_image_folder.0)
        .expect("block_image_folder be loaded");

    for handle in loaded_folder.handles.iter() {
        let id = handle.id().typed_unchecked::<Image>();
        let Some(image) = images.get(id) else {
            warn!(
                "{:?} did not resolve to an `Image` asset.",
                handle.path().unwrap()
            );
            continue;
        };
        loaded_images.push(image);
    }

    if loaded_images.len() == 0 {
        panic!("no images loaded!");
    }

    info!("loaded {} images", loaded_images.len());

    let model = loaded_images[0];
    info!(
        "first image is used as model: width: {}, height: {}",
        model.width(),
        model.height()
    );
    let mut array_texture = Image::new(
        Extent3d {
            width: model.width(),
            height: model.height(),
            depth_or_array_layers: loaded_images.len() as u32,
        },
        TextureDimension::D2,
        loaded_images
            .into_iter()
            .flat_map(|i| i.data.clone())
            .collect(),
        model.texture_descriptor.format,
        RenderAssetUsages::RENDER_WORLD,
    );
    array_texture.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        ..Default::default()
    });
    let handle = images.add(array_texture);
    let handle = materials.add(ArrayTextureMaterial {
        array_texture: handle,
    });
    cmd.insert_resource(MaterialStorage(handle));
    next_state.set(AppState::Run);
}

#[derive(Debug, Resource)]
struct MaterialStorage(Handle<ArrayTextureMaterial>);

#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
struct ArrayTextureMaterial {
    #[texture(0, dimension = "2d_array")]
    #[sampler(1)]
    array_texture: Handle<Image>,
}

impl Material for ArrayTextureMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/chunk.wgsl".into()
    }

    fn vertex_shader() -> ShaderRef {
        "shaders/chunk.wgsl".into()
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline<Self>,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        layout: &bevy::render::mesh::MeshVertexBufferLayout,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        let vertex_layout = layout.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            ATTRIBUTE_DATA.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

/// Basic voxel type with one byte of texture layers
#[derive(Default, Clone, Copy)]
struct BoolVoxel(u8);

impl BoolVoxel {
    pub const EMPTY: Self = BoolVoxel(0);
    pub const GRASS: Self = BoolVoxel(1);
    pub const SOIL: Self = BoolVoxel(2);
    pub const SNOW: Self = BoolVoxel(3);
}

impl MergeVoxel for BoolVoxel {
    type MergeValue = u8;

    fn merge_value(&self) -> Self::MergeValue {
        self.0
    }
}

impl Voxel for BoolVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        if self.0 > 0 {
            VoxelVisibility::Opaque
        } else {
            VoxelVisibility::Empty
        }
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    material_storage: ResMut<MaterialStorage>,
) {
    debug!("setup");

    type SampleShape = ConstShape3u32<22, 22, 22>;

    // Just a solid cube of voxels. We only fill the interior since we need some empty voxels to form a boundary for the mesh.
    // 这是一堆立方体体素组成的实心立方体。我们只填充内部，因为我们需要一些空的体素来形成网格的边界。
    let mut voxels = [BoolVoxel::EMPTY; SampleShape::SIZE as usize];
    // 这里用一串数据表示一个立体的空间
    for z in 1..21 {
        for y in 1..21 {
            for x in 1..21 {
                let i = SampleShape::linearize([x, y, z]);
                if ((x * x + y * y + z * z) as f32).sqrt() < 20.0 {
                    if y < 5 {
                        voxels[i as usize] = BoolVoxel::SOIL;
                    } else if y < 10 {
                        voxels[i as usize] = BoolVoxel::GRASS;
                    } else {
                        voxels[i as usize] = BoolVoxel::SNOW;
                    }
                }
            }
        }
    }
    // 21 x 21 x 21

    let faces = RIGHT_HANDED_Y_UP_CONFIG.faces;

    let mut buffer = GreedyQuadsBuffer::new(voxels.len());
    greedy_quads(
        &voxels,
        &SampleShape {},
        [0; 3],
        [21; 3],
        &faces,
        &mut buffer,
    );
    let num_indices = buffer.quads.num_quads() * 6;
    let num_vertices = buffer.quads.num_quads() * 4;
    let mut indices = Vec::with_capacity(num_indices);
    let mut positions = Vec::with_capacity(num_vertices);
    let mut normals = Vec::with_capacity(num_vertices);
    let mut tex_coords = Vec::with_capacity(num_vertices);

    let mut data = Vec::with_capacity(num_vertices);

    for (block_face_normal_index, (group, face)) in buffer
        .quads
        .groups
        .as_ref()
        .into_iter()
        .zip(faces.into_iter())
        .enumerate()
    {
        for quad in group.into_iter() {
            indices.extend_from_slice(&face.quad_mesh_indices(positions.len() as u32));
            positions.extend_from_slice(&face.quad_mesh_positions(&quad, 1.0));
            normals.extend_from_slice(&face.quad_mesh_normals());
            tex_coords.extend_from_slice(&face.tex_coords(
                RIGHT_HANDED_Y_UP_CONFIG.u_flip_face,
                true,
                &quad,
            ));

            // 计算出 data
            // let a: [u32; 3] = quad.minimum.map(|x| x - 1);
            let a = quad.minimum;
            let index = SampleShape::linearize(a);
            let aa = voxels[index as usize].0;
            let c = (aa - 1) as u32;
            let d = (block_face_normal_index as u32) << 8u32;
            data.extend_from_slice(&[d | c; 4]);
            // data.extend_from_slice(&[(block_face_normal_index as u32) << 8u32 | c; 4],);
            // &[voxels[index as usize].0 as u32; 4],);
        }
    }

    let mut render_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );

    render_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    render_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    render_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, tex_coords);
    render_mesh.insert_attribute(ATTRIBUTE_DATA, VertexAttributeValues::Uint32(data));
    render_mesh.insert_indices(Indices::U32(indices));

    commands.spawn((
        MaterialMeshBundle {
            mesh: meshes.add(render_mesh),
            material: material_storage.0.clone(),
            transform: Transform::from_translation(Vec3::splat(-10.0)),
            ..Default::default()
        },
        Wireframe,
        ShowAabbGizmo {
            color: Some(Color::BLACK),
        },
    ));

    commands.spawn(PointLightBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 5.0, 0.0)),
        point_light: PointLight {
            range: 200.0,
            ..Default::default()
        },
        ..Default::default()
    });
    commands.insert_resource(AmbientLight {
        brightness: lux::OVERCAST_DAY,
        ..default()
    });
}
