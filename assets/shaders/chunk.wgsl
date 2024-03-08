#import bevy_pbr::{
    //forward_io::VertexOutput,
    mesh_view_bindings::view,
    pbr_types::{STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT, PbrInput, pbr_input_new},
    pbr_functions as fns,
    mesh_functions::{get_model_matrix, mesh_position_local_to_clip, mesh_position_local_to_world, mesh_normal_local_to_world},
}
#import bevy_core_pipeline::tonemapping::tone_mapping
#import bevy_pbr::view_transformations::position_world_to_clip;

var<private> VOXEL_NORMALS: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(-1., 0., 0.),
    vec3<f32>(0., -1., 0.),
    vec3<f32>(0., 0., -1.), 
    vec3<f32>(1., 0., 0.), 
    vec3<f32>(0., 1., 0.), 
    vec3<f32>(0., 0., 1.), 
);

// Extracts the normal face index from the encoded voxel data
fn voxel_data_extract_normal(voxel_data: u32) -> vec3<f32> {
    return VOXEL_NORMALS[voxel_data >> 8u & 7u];
}

// Extracts the material index from the encoded voxel data
fn voxel_data_extract_material_index(voxel_data: u32) -> u32 {
    return voxel_data & 255u;
}

@group(2) @binding(0) var my_array_texture: texture_2d_array<f32>;
@group(2) @binding(1) var my_array_texture_sampler: sampler;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) voxel_data: u32,
    @location(2) uv:vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) voxel_data: u32,
    @location(2) world_position: vec4<f32>,
    @location(3) uv:vec2<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let model_matrix = get_model_matrix(vertex.instance_index);
    let world_position = mesh_position_local_to_world(model_matrix, vec4<f32>(vertex.position, 1.0));
    var out: VertexOutput;
    out.world_position = world_position; 
    out.position = mesh_position_local_to_clip(model_matrix, vec4<f32>(vertex.position, 1.0));
    out.world_normal = voxel_data_extract_normal(vertex.voxel_data);
    out.voxel_data = vertex.voxel_data;
    out.uv = vertex.uv;
    return out;
}


struct FragmentInput {
    @builtin(front_facing) is_front: bool,
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) voxel_normal: vec3<f32>,
    /// The voxel data.
    @location(1) voxel_data: u32,
    /// The world position of the voxel vertex.
    @location(2) world_position: vec3<f32>,
    // #import bevy_pbr::mesh_vertex_output
    @location(3) uv:vec2<f32>,
};

@fragment
fn fragment(
    in: FragmentInput
) -> @location(0) vec4<f32> {
    let layer = i32(voxel_data_extract_material_index(in.voxel_data));
    // Prepare a 'processed' StandardMaterial by sampling all textures to resolve
    // the material members
    var pbr_input: PbrInput = pbr_input_new();

    pbr_input.material.base_color = textureSample(my_array_texture, my_array_texture_sampler, in.uv, layer);

    pbr_input.frag_coord = in.frag_coord;
    pbr_input.world_position =  vec4<f32>(in.world_position, 1.0);
    pbr_input.world_normal = (f32(in.is_front) * 2.0 - 1.0) 
        * in.voxel_normal;

    pbr_input.is_orthographic = view.projection[3].w == 1.0;

    pbr_input.N = normalize(in.voxel_normal);
    pbr_input.V = fns::calculate_view(vec4<f32>(in.world_position, 1.0), pbr_input.is_orthographic);
    
    return tone_mapping(fns::apply_pbr_lighting(pbr_input), view.color_grading);
}
