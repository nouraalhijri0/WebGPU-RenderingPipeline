#include "webgpu.h"
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// You can use glm or using your implementation in previous assignments for calculation
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext.hpp>
#define GLg_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform2.hpp>
using namespace std;

//#include "rasterization/Camera.h"
#include "rasterization/Mesh.h"
#include <thread>         
#include <chrono>  

//----------- WEBGPU variables ---------------------------------------
WGPUDevice device;
WGPUQueue queue;
WGPUSwapChain swapchain;
WGPURenderPipeline pipeline;
WGPUBuffer vertBuf; // vertex buffer
WGPUBuffer indxBuf; // index buffer
WGPUBuffer uRotBuf; // rotation buffer
WGPUBuffer uCameraBuf;
WGPUBuffer uNormalBuf;
WGPUBuffer uLightBuf;
WGPUBindGroup bindGroup;

WGPUTexture tex;
WGPUSampler samplerTex;
WGPUTextureView texView;
WGPUExtent3D texSize = {};
WGPUTextureDescriptor texDesc = {};
WGPUTextureDataLayout texDataLayout = {};
WGPUImageCopyTexture texCopy = {};

WGPUTexture depthTex;
WGPUSampler depthSamplerTex;
WGPUTextureView depthTexView;
WGPUExtent3D depthTexSize = {};
WGPUTextureDescriptor depthTexDesc = {};
WGPUTextureDataLayout depthTexDataLayout = {};

//----------- Camera and viewport initializations ---------------------------------------
const int viewWidth = 1024, viewHeight = 768;
glm::vec3 cameraPos(0.0f, 0.0f, -2.0f);
glm::vec3 target(0.0f, 0.0f, 0.0f);
glm::vec3 up(0.0f, 1.0f, 0.0f);
glm::mat4 projection = glm::perspective(45.0f, (float)viewWidth / (float)viewHeight, 0.1f, 100.0f);
glm::mat4 camera = glm::lookAt(cameraPos, target, up);
glm::mat4 translate = glm::translate(glm::vec3(0.0, 0.0, 0.0));//(glm::vec3(0.217, -1.575, 0.0));
glm::mat4 model = translate;
glm::mat4 MVP = projection * camera * model;
glm::mat4 MV = camera * model;
glm::mat4 NormalMatrix = glm::transpose(glm::inverse(camera * model));
struct Camera {
	glm::mat4 mv;
	glm::mat4 mvp;
	glm::vec3 position;
	glm::mat4 mvpInverse;
	float padding = 0.0f;
};
Camera perspectiveCamera;
void setCameraBufferVariables() {
	perspectiveCamera.mv = MV;
	perspectiveCamera.mvp = MVP;
	perspectiveCamera.position = glm::vec3(4, 3, 3);
}


//----------- Lights initializations ---------------------------------------
struct Light {
	glm::vec3 position;
	float padding_pos = 0.0f;
	glm::vec3 color;
	float padding_color = 0.0f;
	glm::mat4 Translate;
	glm::mat4 TranslateInv;
	glm::mat4 instanceMatrices[2];
	float angleOfRotation;
};
Light light1;
void setLightBufferVariables() {
	light1.position = glm::vec3(200.0f, 200.0f, 0.0f);
	light1.color = glm::vec3(1.0f, 1.0f, 1.0f);
	light1.Translate = glm::mat4(1.0f);
	const int xCount = 2;
	const int yCount = 1;
	const int numInstances = xCount * yCount;
	const int matrixFloatCount = 16;
	int matrixSize = 4 * matrixFloatCount;
	int uniformBufferSize = numInstances * matrixSize;
	glm::mat4 modelMatrices[numInstances];
	int indexMat = 0;
	int step = 10.0;
	for (int x = 0; x < xCount; x++) {
		for (int y = 0; y < yCount; y++) {

			glm::mat4 tran = glm::translate(glm::mat4(1.0), glm::vec3(step * (x - xCount / 2 + 0.5), step * (y - yCount / 2 + 0.5), 0.0f));
			light1.instanceMatrices[indexMat] = tran;

			indexMat++;
		}
	}
	light1.TranslateInv = glm::transpose(glm::inverse(light1.Translate));
}

Mesh object;


/**
 * Current rotation angle (in degrees, updated per frame).
 */
float rotDeg = 0.5f;
const float PI = 3.141592653589793;
vector <glm::vec3> points;
vector <glm::vec3> colors;
vector <glm::vec2>textures;
std::string toon_vert_wgsl;
std::string toon_frag_wgsl;
std::string gouraud_vert_wgsl;
std::string gouraud_frag_wgsl;
std::string phong_vert_wgsl;
std::string phong_frag_wgsl;
std::string texture_vert_wgsl;
std::string texture_frag_wgsl;
std::string glass_vert_wgsl;
std::string glass_frag_wgsl;
int example;
bool phong = false, gouraud = false, toon = false, sphericalTex = false, cylindricalTex = false, bumpmap = false, sphere = false, teapot = false;
// This is just an example vertex data
void setVertexData()
{

	vector<glm::vec3> MeshPoints = object.points;
	vector<glm::vec3> MeshNormals = object.normals;
	vector<glm::vec2> VertTexture;
	for (int i = 0; i < MeshPoints.size(); i++) {
		//compute vertex and normal 3D , uv 2D
		glm::vec3 point = MeshPoints[i];
		glm::vec3 normal_ = MeshNormals[i];
		float u, v, no = 0.65;
		if (sphericalTex) {
			u = 0.5 + (atan2(point.x / no, point.z / no) / (PI * 2.0));
			v = 0.5 - (asin(point.y / no) / PI);
		}
		else if (cylindricalTex) {
			u = 0.5 + (atan2(point.x / no, point.z / no) / (PI * 2.0));
			v = 0.5 - ((point.y / no) / 2.0);
		} 

		glm::vec2 texture = glm::vec2(u, v);
		points.emplace_back(glm::vec3(point.x, point.y, point.z));
		colors.emplace_back(glm::vec3(normal_.x, normal_.y, normal_.z));
		textures.emplace_back(glm::vec2(texture.x, texture.y));
	}
}

static void setupShader()
{
	//toon shader
	toon_vert_wgsl = R"(
		
		[[block]]
		struct VertexIn {
			[[location(0)]] aPos : vec3<f32>;
			[[location(1)]] aNormal : vec3<f32>;
		};
		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] vNormal : vec3<f32>;
			[[location(1)]] vPosition : vec3<f32>;
		};

		[[block]]
		struct Camera {
			mv: mat4x4<f32>;
			mvp: mat4x4<f32>; 
			position: vec3<f32>;
		};
		[[group(0), binding(0)]] var<uniform> uCamera : Camera;

		[[block]]
		struct NormalMatrix {
			value : mat4x4<f32>;
		};
		[[group(0), binding(1)]] var<uniform> uNormalMatrix: NormalMatrix;


		[[block]]
		struct Light {
			position : vec3<f32>;
			color : vec3<f32>;
			Translate: mat4x4<f32>;
			TranslateInv: mat4x4<f32>;
		};
		[[group(0), binding(2)]] var<uniform> uLight : Light;
		let PI : f32 = 3.141592653589793;
		fn radians(degs : f32) -> f32 {
			return (degs * PI) / 180.0;
		}

		[[block]]
		struct Rotation {
			degs: f32;
		};
		[[group(0), binding(3)]] var<uniform> uRot : Rotation;

		[[stage(vertex)]]
		fn main(input : VertexIn) -> VertexOut {

			var output : VertexOut;
			var rads : f32 = radians(uRot.degs);
			var cosA : f32 = cos(rads);
			var sinA : f32 = sin(rads);
			var rot : mat4x4<f32> = mat4x4<f32>(
				vec4<f32>(cosA, 0.0, sinA, 0.0),
				vec4<f32>( 0.0, 1.0, 0.0, 0.0),
				vec4<f32>( -sinA, 0.0,cosA, 0.0), 
				vec4<f32>( 0.0, 0.0, 0.0, 1.0));
			
			output.Position =  uCamera.mvp  * uLight.Translate *  vec4<f32>(input.aPos, 1.0);
			output.vNormal =  input.aNormal; 
			output.vPosition = input.aPos;
			return output; 
		}
	)";

	toon_frag_wgsl = R"(
		[[block]]
		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] vNormal : vec3<f32>;
			[[location(1)]] vPosition : vec3<f32>;
		};
		[[block]]
		struct Camera {
			mv: mat4x4<f32>;
			mvp: mat4x4<f32>; 
			position: vec3<f32>;
		};
		[[group(0), binding(0)]] var<uniform> uCamera : Camera;

		
		[[block]]
		struct NormalMatrix {
			value : mat4x4<f32>;
		};
		[[group(0), binding(1)]] var<uniform> uNormalMatrix: NormalMatrix;


		[[block]]
		struct Light {
			position: vec3<f32>;
			color: vec3<f32>;
		};
		[[group(0), binding(2)]] var<uniform> uLight : Light;
		
		[[block]]
		struct Rotation {
			degs: f32;
		};
		[[group(0), binding(3)]] var<uniform> uRot : Rotation;
		
		[[stage(fragment)]]
		fn main(in : VertexOut) -> [[location(0)]] vec4<f32> {
			let Ka= vec3<f32>(0.75,0.0,0.5);
			let Kd = vec3<f32>(0.4, 0.0, 0.75);
			let VertexPosition = (uCamera.mvp *vec4<f32>(in.vPosition, 1.0)).xyz;
			let VertexNormal =normalize((uNormalMatrix.value *  vec4<f32>(in.vNormal, 1.0)).xyz);
			let Lightpos = (uCamera.mv * vec4<f32>(uLight.position, 1.0)).xyz; 
			let Lightcolor = uLight.color;  
			let light_dir = normalize(Lightpos - VertexPosition);
			let diffuse_strength = max(dot(VertexNormal, light_dir), 0.0);
			let finalColor = (uLight.color * (Ka + Kd * floor(diffuse_strength * 4.0 ) * 0.25));
			return vec4<f32>(finalColor, 1.0);
		}
	)";

	//gouraud shader
	gouraud_vert_wgsl = R"(
		[[block]]
		struct VertexIn {
			[[location(0)]] aPos : vec3<f32>;
			[[location(1)]] aNormal : vec3<f32>;
			[[location(2)]] aTex : vec2<f32>;
		};

		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] vColor : vec3<f32>;
		
		};

		[[block]]
		struct Camera {
			mv: mat4x4<f32>;
			mvp: mat4x4<f32>; 
			position: vec3<f32>;
		};
		[[group(0), binding(0)]] var<uniform> uCamera : Camera;

		[[block]]
		struct NormalMatrix {
			value: mat4x4<f32>;
		};
		[[group(0), binding(1)]] var<uniform> uNormalMatrix : NormalMatrix;

		[[block]]
		struct Light {
			position : vec3<f32>;
			color : vec3<f32>;
			Translate: mat4x4<f32>;
			TranslateInv: mat4x4<f32>;
			};
		[[group(0), binding(2)]] var<uniform> uLight : Light;

		let Ka= vec3<f32>(1.0,0.0,0.5);
		let Kd = vec3<f32>(1.0, 1.0, 1.0);
		let Ks = vec3<f32>(0.0, 0.5, 0.0);
		let shininess = 76.8;

		let PI : f32 = 3.141592653589793;
		fn radians(degs : f32) -> f32 {
			return (degs * PI) / 180.0;
		}

		[[block]]
		struct Rotation {
			degs : f32;
		};

		[[group(0), binding(3)]] var<uniform> uRot : Rotation;

		[[stage(vertex)]]
		fn main(input : VertexIn) -> VertexOut {
			var rads : f32 = radians(uRot.degs);
			var cosA : f32 = cos(rads);
			var sinA : f32 = sin(rads);
			var rot : mat4x4<f32> = mat4x4<f32>(
				vec4<f32>(cosA, 0.0, sinA, 0.0),
				vec4<f32>( 0.0, 1.0, 0.0, 0.0),
				vec4<f32>( -sinA, 0.0,cosA, 0.0), 
				vec4<f32>( 0.0, 0.0, 0.0, 1.0));

			var output : VertexOut;
			output.Position =  uCamera.mvp  *rot * uLight.Translate *  vec4<f32>(input.aPos, 1.0);
			let VertexPosition = (uCamera.mv  * rot * uLight.Translate * vec4<f32>(input.aPos, 1.0)).xyz;
			let VertexNormal =normalize((   uNormalMatrix.value * rot *  uLight.TranslateInv * vec4<f32>(input.aNormal, 1.0)).xyz);
			let Lightpos =   (uCamera.mv * vec4<f32>(uLight.position, 1.0)).xyz; 
			let Lightcolor = uLight.color;  
			let light_dir = normalize(Lightpos - VertexPosition); 
			let view_dir = normalize(VertexPosition);
			let h = normalize(-view_dir + light_dir);
			output.vColor = (Lightcolor * (Ka + Kd * max(dot(VertexNormal, light_dir), 0.0)+ Ks * pow(max(dot(h, VertexNormal), 0.0), 80.0)));  
			return output;
		}
	)";

	gouraud_frag_wgsl = R"(
		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] vColor : vec3<f32>;
		};

		[[group(0), binding(4)]]
		var t_diffuse: texture_2d<f32>;
		[[group(0), binding(5)]]
		var s_diffuse: sampler;

		[[stage(fragment)]]
		fn main(in : VertexOut) -> [[location(0)]] vec4<f32> {
			return vec4<f32>(in.vColor, 1.0);
		}
	)";

	//Phong shader
	phong_vert_wgsl = R"(
		[[block]]
		struct VertexIn {
			[[builtin(instance_index)]] instanceIdx : u32;
			[[location(0)]] aPos : vec3<f32>;
			[[location(1)]] aNormal : vec3<f32>;
			[[location(2)]] aTex : vec2<f32>;
		};

		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] vColor : vec3<f32>;
			[[location(1)]] vTexCoor : vec2<f32>;
			[[location(2)]] vNormal : vec3<f32>;
			[[location(3)]] vPosition : vec3<f32>;
			[[location(4)]] vLightPosition : vec3<f32>;
			[[location(5)]] vLightColor : vec3<f32>;
		};

		[[block]]
		struct Camera {
			mv: mat4x4<f32>;
			mvp: mat4x4<f32>; 
			position: vec3<f32>;
		};
		[[group(0), binding(0)]] var<uniform> uCamera : Camera;

		[[block]]
		struct NormalMatrix {
			value: mat4x4<f32>;
		};
		[[group(0), binding(1)]] var<uniform> uNormalMatrix : NormalMatrix;

		[[block]]
		struct Light {
			position : vec3<f32>;
			color : vec3<f32>;
			Translate: mat4x4<f32>;
			TranslateInv: mat4x4<f32>;
			instanceMatrices : [[stride(64)]] array<mat4x4<f32>, 2>;
			};
		[[group(0), binding(2)]] var<uniform> uLight : Light;	

		let PI : f32 = 3.141592653589793;
		fn radians(degs : f32) -> f32 {
			return (degs * PI) / 180.0;
		}

		[[block]]
		struct Rotation {
			degs : f32;
		};
		[[group(0), binding(3)]] var<uniform> uRot : Rotation;

		[[stage(vertex)]]
		fn main(input : VertexIn) -> VertexOut {
			var rads : f32 = radians(uRot.degs);
			var cosA : f32 = cos(rads);
			var sinA : f32 = sin(rads);
			var rot : mat4x4<f32> = mat4x4<f32>(
				vec4<f32>(cosA, 0.0, sinA, 0.0),
				vec4<f32>( 0.0, 1.0, 0.0, 0.0),
				vec4<f32>( -sinA, 0.0,cosA, 0.0), 
				vec4<f32>( 0.0, 0.0, 0.0, 1.0));
			var output : VertexOut;
			output.vColor = vec3<f32>(1.0, 0.0, 0.0);
			output.vPosition = (uCamera.mv * uLight.instanceMatrices[input.instanceIdx] * vec4<f32>(input.aPos, 1.0)).xyz;
			output.vNormal =  normalize((uNormalMatrix.value * vec4<f32>(input.aNormal, 1.0)).xyz);
			output.vLightPosition = (uCamera.mv * vec4<f32>(uLight.position, 1.0)).xyz;
			output.vLightColor = uLight.color;
			output.Position = uCamera.mvp * uLight.instanceMatrices[input.instanceIdx] *vec4<f32>(input.aPos, 1.0);
			return output;
			}
		)";
	phong_frag_wgsl = R"(
		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] vColor : vec3<f32>;
			[[location(1)]] vTexCoor : vec2<f32>;
			[[location(2)]] vNormal : vec3<f32>;
			[[location(3)]] vPosition : vec3<f32>;
			[[location(4)]] vLightPosition : vec3<f32>;
			[[location(5)]] vLightColor : vec3<f32>;
		};

		[[group(0), binding(4)]]
		var t_diffuse: texture_2d<f32>;
		[[group(0), binding(5)]]
		var s_diffuse: sampler;

		//Material parametes emerals
		let Ka= vec3<f32>(1.0,0.0,0.5);
		let Kd = vec3<f32>(1.0, 1.0, 1.0);
		let Ks = vec3<f32>(0.0, 0.5, 0.0);
		let shininess = 76.8;

		fn ads(in : VertexOut) -> vec3<f32> {

			let light_dir = normalize(in.vLightPosition - in.vPosition);
			let view_dir = -normalize(in.vPosition);
			let half_dir = normalize(view_dir + light_dir);

			let diffuse_strength = max(dot(in.vNormal, light_dir), 0.0);
			let specular_strength : f32 = pow(max(dot(half_dir, in.vNormal), 0.0), shininess);

			return (in.vLightColor * (Ka + Kd * diffuse_strength + Ks * specular_strength));
		}

		[[stage(fragment)]]
		fn main(in : VertexOut) -> [[location(0)]] vec4<f32> {
			return vec4<f32>(ads(in), 1.0);
		}
	)";

	//texture shader for sphere
	texture_vert_wgsl = R"(
		[[block]]
		struct VertexIn {
			[[location(0)]] aPos : vec3<f32>;
			[[location(1)]] aNormal : vec3<f32>;
			[[location(2)]] aTex : vec2<f32>;
		};
		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] vTexCoor : vec2<f32>;
		};
		[[block]]
		struct Camera {
			mv: mat4x4<f32>;
			mvp: mat4x4<f32>; 
			position: vec3<f32>;
		};
		[[group(0), binding(0)]] var<uniform> uCamera : Camera;

		[[block]]
		struct NormalMatrix {
			value: mat4x4<f32>;
		};
		[[group(0), binding(1)]] var<uniform> uNormalMatrix : NormalMatrix;

		[[block]]
		struct Light {
			position : vec3<f32>;
			color : vec3<f32>;
			Translate: mat4x4<f32>;
			TranslateInv: mat4x4<f32>;
			};
		[[group(0), binding(2)]] var<uniform> uLight : Light;

		let PI : f32 = 3.141592653589793;
		fn radians(degs : f32) -> f32 {
			return (degs * PI) / 180.0;
		}

		[[block]]
		struct Rotation {
			degs : f32;
		};

		[[group(0), binding(3)]] var<uniform> uRot : Rotation;

		[[stage(vertex)]]
		fn main(input : VertexIn) -> VertexOut {
			var rads : f32 = radians(uRot.degs);
			var cosA : f32 = cos(rads);
			var sinA : f32 = sin(rads);
			var rot : mat4x4<f32> = mat4x4<f32>(
				vec4<f32>(cosA, 0.0, sinA, 0.0),
				vec4<f32>( 0.0, 1.0, 0.0, 0.0),
				vec4<f32>( -sinA, 0.0,cosA, 0.0), 
				vec4<f32>( 0.0, 0.0, 0.0, 1.0));
			var output : VertexOut;
			output.Position =  uCamera.mvp  *rot * uLight.Translate *  vec4<f32>(input.aPos, 1.0);
			output.vTexCoor = input.aTex;
			return output;
		}
	)";
	
	texture_frag_wgsl = R"(
		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] vTexCoor : vec2<f32>;
		};

		[[group(0), binding(4)]]
		var t_diffuse: texture_2d<f32>;
		[[group(0), binding(5)]]
		var s_diffuse: sampler;

		[[stage(fragment)]]
		fn main(in : VertexOut) -> [[location(0)]] vec4<f32> {
			return textureSample(t_diffuse, s_diffuse, in.vTexCoor);
		}
	)";

	//glass shading
	glass_vert_wgsl = R"(
		[[block]]
		struct VertexIn {
			[[location(0)]] aPos : vec3<f32>;
			[[location(1)]] aNormal : vec3<f32>;
			[[location(2)]] aTex : vec2<f32>;
		};

		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] v_reflection : vec3<f32>;
			[[location(1)]] v_refraction : vec3<f32>;
			[[location(2)]] v_fresnel : vec3<f32>;
		
		};

		[[block]]
		struct Camera {
			mv: mat4x4<f32>;
			mvp: mat4x4<f32>; 
			position: vec3<f32>;
		};
		[[group(0), binding(0)]] var<uniform> uCamera : Camera;

		[[block]]
		struct NormalMatrix {
			value: mat4x4<f32>;
		};
		[[group(0), binding(1)]] var<uniform> uNormalMatrix : NormalMatrix;

		[[block]]
		struct Light {
			position : vec3<f32>;
			color : vec3<f32>;
			Translate: mat4x4<f32>;
			TranslateInv: mat4x4<f32>;
			};
		[[group(0), binding(2)]] var<uniform> uLight : Light;
		
		let air = 1.0;
		let glass = 1.51714;
		var eta : f32 = (air / glass);
		let r0 = ((air - glass) * (air - glass)) / ((air + glass) * (air + glass));

		let PI : f32 = 3.141592653589793;
		fn radians(degs : f32) -> f32 {
			return (degs * PI) / 180.0;
		}

		[[block]]
		struct Rotation {
			degs : f32;
		};

		[[group(0), binding(3)]] var<uniform> uRot : Rotation;

		[[stage(vertex)]]
		fn main(input : VertexIn) -> VertexOut {
			var rads : f32 = radians(uRot.degs);
			var cosA : f32 = cos(rads);
			var sinA : f32 = sin(rads);
			var rot : mat4x4<f32> = mat4x4<f32>(
				vec4<f32>(cosA, 0.0, sinA, 0.0),
				vec4<f32>( 0.0, 1.0, 0.0, 0.0),
				vec4<f32>( -sinA, 0.0,cosA, 0.0), 
				vec4<f32>( 0.0, 0.0, 0.0, 1.0));

			var output : VertexOut;
			output.Position =  uCamera.mvp  *rot * uLight.Translate *  vec4<f32>(input.aPos, 1.0);
			let VertexPosition = (uCamera.mv  * rot * uLight.Translate * vec4<f32>(input.aPos, 1.0)).xyz;
			var incident : vec3<f32> = normalize(VertexPosition - uCamera.position);
			let VertexNormal =normalize((   uNormalMatrix.value * rot *  uLight.TranslateInv * vec4<f32>(input.aNormal, 1.0)).xyz);
			output.v_refraction = refract(incident, VertexNormal, eta);
			output.v_reflection = reflect(incident, VertexNormal);
			output.v_fresnel = r0 + (1.0 - r0) * pow((1.0 - dot(-incident, VertexNormal)), 5.0);
			return output;
		}
	)";

	glass_frag_wgsl = R"(
		struct VertexOut {
			[[builtin(position)]] Position : vec4<f32>;
			[[location(0)]] v_reflection : vec3<f32>;
			[[location(1)]] v_refraction : vec3<f32>;
			[[location(2)]] v_fresnel : vec3<f32>;
		
		};

		[[group(0), binding(4)]]
		var t_diffuse: texture_2d<f32>;
		[[group(0), binding(5)]]
		var s_diffuse: sampler;

		[[stage(fragment)]]
		fn main(in : VertexOut) -> [[location(0)]] vec4<f32> {
			var refractionColor : vec4<f32> = texture(u_cubemap, normalize(v_refraction));
			var reflectionColor : vec4<f32> = texture(u_cubemap, normalize(v_reflection));
			return mix(refractionColor, reflectionColor, v_fresnel);
		}
	)";
}


/**
 * Helper to create a shader from WGSL source.
 *
 * \param[in] code WGSL shader source
 * \param[in] label optional shader name
 */
static WGPUShaderModule createShader(const char* const code, const char* label = nullptr) {
	WGPUShaderModuleWGSLDescriptor wgsl = {};
	wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
	wgsl.source = code;
	WGPUShaderModuleDescriptor desc = {};
	desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl);
	desc.label = label;
	return wgpuDeviceCreateShaderModule(device, &desc);
}

/**
 * Helper to create a buffer.
 *
 * \param[in] data pointer to the start of the raw data
 * \param[in] size number of bytes in \a data
 * \param[in] usage type of buffer
 */
static WGPUBuffer createBuffer(const void* data, size_t size, WGPUBufferUsage usage) {
	WGPUBufferDescriptor desc = {};
	desc.usage = WGPUBufferUsage_CopyDst | usage;
	desc.size = size;
	WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
	wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
	return buffer;
}

/**
 * Helper to create a texture.
 *
 * \param[in]
 * \param[in]
 * \param[in]
 */
 //create texture
int imgWidth = 1440, imgHeight = 720, nrChannels = 4;
//unsigned char* img;
unsigned char* img= new unsigned char[imgWidth * imgHeight * 4];
static WGPUTexture createTexture(unsigned char* data, unsigned int w, unsigned int h) {
	if (img == nullptr)
		throw(std::string("Failed to load texture"));

	texSize.depthOrArrayLayers = 1;
	texSize.height = h;
	texSize.width = w;
	texDesc.sampleCount = 1;
	texDesc.mipLevelCount = 1;
	texDesc.dimension = WGPUTextureDimension_2D;
	texDesc.size = texSize;
	texDesc.usage = WGPUTextureUsage_Sampled | WGPUTextureUsage_CopyDst;
	texDesc.format = WGPUTextureFormat_RGBA8Unorm;
	texDataLayout.offset = 0;
	texDataLayout.bytesPerRow = 4 * w;
	texDataLayout.rowsPerImage = h;
	texCopy.texture = wgpuDeviceCreateTexture(device, &texDesc);
	wgpuQueueWriteTexture(queue, &texCopy, data, w * h * 4, &texDataLayout, &texSize);
	return texCopy.texture;
}

static void createDepthTexture() {
	depthTexSize.width = viewWidth;
	depthTexSize.height = viewHeight;
	depthTexSize.depthOrArrayLayers = 1;
	depthTexDesc.sampleCount = 1;
	depthTexDesc.mipLevelCount = 1;
	depthTexDesc.dimension = WGPUTextureDimension_2D;
	depthTexDesc.size = depthTexSize;
	depthTexDesc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
	depthTexDesc.format = WGPUTextureFormat_Depth32Float;
	depthTex = wgpuDeviceCreateTexture(device, &depthTexDesc);
	WGPUTextureViewDescriptor texViewDesc = {};
	texViewDesc.dimension = WGPUTextureViewDimension_2D;
	texViewDesc.format = WGPUTextureFormat_Depth32Float;
	depthTexView = wgpuTextureCreateView(depthTex, &texViewDesc);
	WGPUSamplerDescriptor sampleDesc = {};
	sampleDesc.addressModeU = WGPUAddressMode_ClampToEdge;
	sampleDesc.addressModeV = WGPUAddressMode_ClampToEdge;
	sampleDesc.addressModeW = WGPUAddressMode_ClampToEdge;
	sampleDesc.magFilter = WGPUFilterMode_Linear;
	sampleDesc.minFilter = WGPUFilterMode_Linear;
	sampleDesc.mipmapFilter = WGPUFilterMode_Nearest;
	sampleDesc.lodMaxClamp = 100.0;
	sampleDesc.lodMinClamp = 0.0;
	sampleDesc.compare = WGPUCompareFunction_LessEqual;
	sampleDesc.maxAnisotropy = 1;
	depthSamplerTex = wgpuDeviceCreateSampler(device, &sampleDesc);
}

/**
 * Bare minimum pipeline to draw a triangle using the above shaders.
 */
static void createPipelineAndBuffers() {
	// compile shaders
	// NOTE: these are now the WGSL shaders (tested with Dawn and Chrome Canary)

	setupShader();

	//Here you should specify your shaders!!
	WGPUShaderModule vertMod;
	WGPUShaderModule fragMod;
	if (gouraud) {
		vertMod = createShader(gouraud_vert_wgsl.c_str());
		fragMod = createShader(gouraud_frag_wgsl.c_str());
	}
	else if (toon) {
		vertMod = createShader(toon_vert_wgsl.c_str());
		fragMod = createShader(toon_frag_wgsl.c_str());
	}
	else if (phong) {
		vertMod = createShader(phong_vert_wgsl.c_str());
		fragMod = createShader(phong_frag_wgsl.c_str());
	}
	else if (sphericalTex || cylindricalTex || bumpmap) {
		vertMod = createShader(texture_vert_wgsl.c_str());
		fragMod = createShader(texture_frag_wgsl.c_str());
	}

	// bind group layout for camera
	WGPUBufferBindingLayout uniformLayout = {};
	uniformLayout.type = WGPUBufferBindingType_Uniform;
	WGPUBindGroupLayoutEntry uniformEntry = {};
	uniformEntry.binding = 0;
	uniformEntry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
	uniformEntry.buffer = uniformLayout;
	uniformEntry.sampler = { 0 };

	// bind group layout for normal
	WGPUBufferBindingLayout normalLayout = {};
	normalLayout.type = WGPUBufferBindingType_Uniform;
	WGPUBindGroupLayoutEntry normalEntry = {};
	normalEntry.binding = 1;
	normalEntry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
	normalEntry.buffer = normalLayout;
	normalEntry.sampler = { 0 };

	// bind group layout for light
	WGPUBufferBindingLayout lightLayout = {};
	lightLayout.type = WGPUBufferBindingType_Uniform;
	WGPUBindGroupLayoutEntry lightEntry = {};
	lightEntry.binding = 2;
	lightEntry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
	lightEntry.buffer = lightLayout;
	lightEntry.sampler = { 0 };

	// bind group layout for rotation
	WGPUBufferBindingLayout rotationLayout = {};
	rotationLayout.type = WGPUBufferBindingType_Uniform;
	WGPUBindGroupLayoutEntry rotationEntry = {};
	rotationEntry.binding = 3;
	rotationEntry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
	rotationEntry.buffer = rotationLayout;
	rotationEntry.sampler = { 0 };


	// bind group layout for texture
	WGPUTextureBindingLayout texLayout = {};
	texLayout.sampleType = WGPUTextureSampleType_Float;
	texLayout.viewDimension = WGPUTextureViewDimension_2D;
	texLayout.multisampled = false;
	WGPUBindGroupLayoutEntry bglTexEntry = {};
	bglTexEntry.binding = 4;
	bglTexEntry.visibility = WGPUShaderStage_Fragment;
	bglTexEntry.texture = texLayout;

	// bind group layout for sampler
	WGPUSamplerBindingLayout samplerLayout = {};
	samplerLayout.type = WGPUSamplerBindingType_Filtering;
	WGPUBindGroupLayoutEntry bglSamplerEntry = {};
	bglSamplerEntry.binding = 5;
	bglSamplerEntry.visibility = WGPUShaderStage_Fragment;
	bglSamplerEntry.sampler = samplerLayout;

	int numOfEntries = 6;
	WGPUBindGroupLayoutEntry* allBgLayoutEntries = new WGPUBindGroupLayoutEntry[numOfEntries];
	allBgLayoutEntries[0] = uniformEntry;
	allBgLayoutEntries[1] = normalEntry;
	allBgLayoutEntries[2] = lightEntry;
	allBgLayoutEntries[3] = rotationEntry;
	allBgLayoutEntries[4] = bglTexEntry;
	allBgLayoutEntries[5] = bglSamplerEntry;

	WGPUBindGroupLayoutDescriptor bglDesc = {};
	bglDesc.entryCount = numOfEntries;
	bglDesc.entries = allBgLayoutEntries;
	WGPUBindGroupLayout bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

	// pipeline layout (used by the render pipeline, released after its creation): remember to add all uniform layout to pipeline layout
	WGPUPipelineLayoutDescriptor layoutDesc = {};
	layoutDesc.bindGroupLayoutCount = 1;
	layoutDesc.bindGroupLayouts = &bindGroupLayout;
	WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);

	tex = createTexture(img, imgWidth, imgHeight);
	WGPUTextureViewDescriptor texViewDesc = {};
	texViewDesc.dimension = WGPUTextureViewDimension_2D;
	texViewDesc.format = WGPUTextureFormat_RGBA8Unorm;
	texView = wgpuTextureCreateView(tex, &texViewDesc);
	createDepthTexture();

	WGPUSamplerDescriptor sampleDesc = {};
	sampleDesc.addressModeU = WGPUAddressMode_ClampToEdge;
	sampleDesc.addressModeV = WGPUAddressMode_ClampToEdge;
	sampleDesc.addressModeW = WGPUAddressMode_ClampToEdge;
	sampleDesc.magFilter = WGPUFilterMode_Linear;
	sampleDesc.minFilter = WGPUFilterMode_Linear;
	sampleDesc.mipmapFilter = WGPUFilterMode_Nearest;
	sampleDesc.lodMaxClamp = 100.0;
	sampleDesc.lodMinClamp = 0.0;
	sampleDesc.compare = WGPUCompareFunction_Undefined;
	sampleDesc.maxAnisotropy = 1;
	samplerTex = wgpuDeviceCreateSampler(device, &sampleDesc);

	// describe vertex buffer layouts: Need to care about the memory layout
	WGPUVertexAttribute vertAttrs[3] = {};
	vertAttrs[0].format = WGPUVertexFormat_Float32x3;
	vertAttrs[0].offset = 0;
	vertAttrs[0].shaderLocation = 0;
	vertAttrs[1].format = WGPUVertexFormat_Float32x3;
	vertAttrs[1].offset = 3 * sizeof(float);
	vertAttrs[1].shaderLocation = 1;
	vertAttrs[2].format = WGPUVertexFormat_Float32x2;
	vertAttrs[2].offset = 6 * sizeof(float);
	vertAttrs[2].shaderLocation = 2;

	WGPUVertexBufferLayout vertexBufferLayout = {};
	vertexBufferLayout.arrayStride = 8 * sizeof(float);
	vertexBufferLayout.attributeCount = 3;
	vertexBufferLayout.attributes = vertAttrs;

	// create the vertex data
	setVertexData();
	float* vertData = new float[points.size() * 8];
	int index = 0;
	int i;

	// Memory layout
	for (i = 0; i < points.size(); i++)
	{
		vertData[index + 0] = float(points[i].x);
		vertData[index + 1] = float(points[i].y);
		vertData[index + 2] = float(points[i].z);
		vertData[index + 3] = float(colors[i].x);
		vertData[index + 4] = float(colors[i].y);
		vertData[index + 5] = float(colors[i].z);
		vertData[index + 6] = float(textures[i].x);
		vertData[index + 7] = float(textures[i].y);
		index += 8;

	}

	// create the indices data: the index to draw each triangle
	uint16_t* indxData = new uint16_t[object.faces.size() * 3];
	for (int i = 0; i < object.faces.size(); i++) {

		indxData[3 * i + 0] = (uint16_t)(object.faces[i][0] - 1);
		indxData[3 * i + 1] = (uint16_t)(object.faces[i][1] - 1);
		indxData[3 * i + 2] = (uint16_t)(object.faces[i][2] - 1);

	}

	vertBuf = createBuffer(vertData, points.size() * 8 * sizeof(float), WGPUBufferUsage_Vertex);
	indxBuf = createBuffer(indxData, object.faces.size() * 3 * sizeof(uint16_t), WGPUBufferUsage_Index);

	//setting up camera buffer
	setCameraBufferVariables();
	uCameraBuf = createBuffer(&perspectiveCamera, sizeof(perspectiveCamera), WGPUBufferUsage_Uniform);
	WGPUBindGroupEntry bgCameraEntry = {};
	bgCameraEntry.binding = 0;
	bgCameraEntry.buffer = uCameraBuf;
	bgCameraEntry.offset = 0;
	bgCameraEntry.size = sizeof(perspectiveCamera);

	//setting up the normal buffer
	uNormalBuf = createBuffer(&NormalMatrix, sizeof(NormalMatrix), WGPUBufferUsage_Uniform);
	WGPUBindGroupEntry bgNormalEntry = {};
	bgNormalEntry.binding = 1;
	bgNormalEntry.buffer = uNormalBuf;
	bgNormalEntry.offset = 0;
	bgNormalEntry.size = sizeof(NormalMatrix);

	//setting up the light buffer
	setLightBufferVariables();
	uLightBuf = createBuffer(&light1, sizeof(light1), WGPUBufferUsage_Uniform);
	WGPUBindGroupEntry bgLightEntry = {};
	bgLightEntry.binding = 2;
	bgLightEntry.buffer = uLightBuf;
	bgLightEntry.offset = 0;
	bgLightEntry.size = sizeof(light1);

	//setting up the rotation buffer
	uRotBuf = createBuffer(&rotDeg, sizeof(rotDeg), WGPUBufferUsage_Uniform);
	WGPUBindGroupEntry bgRotationEntry = {};
	bgRotationEntry.binding = 3;
	bgRotationEntry.buffer = uRotBuf;
	bgRotationEntry.offset = 0;
	bgRotationEntry.size = sizeof(rotDeg);

	//setting up the texture entry
	WGPUBindGroupEntry bgTexEntry = {};
	bgTexEntry.binding = 4;
	bgTexEntry.textureView = texView;

	//setting up the sampler entry
	WGPUBindGroupEntry bgSamplerEntry = {};
	bgSamplerEntry.binding = 5;
	bgSamplerEntry.sampler = samplerTex;

	WGPUBindGroupEntry* allBgEntries = new WGPUBindGroupEntry[numOfEntries];
	allBgEntries[0] = bgCameraEntry;
	allBgEntries[1] = bgNormalEntry;
	allBgEntries[2] = bgLightEntry;
	allBgEntries[3] = bgRotationEntry;
	allBgEntries[4] = bgTexEntry;
	allBgEntries[5] = bgSamplerEntry;

	WGPUBindGroupDescriptor bgDesc = {};
	bgDesc.layout = bindGroupLayout;
	bgDesc.entryCount = numOfEntries;
	bgDesc.entries = allBgEntries;
	bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);


	// last bit of clean-up
	wgpuBindGroupLayoutRelease(bindGroupLayout);

	// Vertex state
	WGPUVertexState vertex = {};
	vertex.module = vertMod;
	vertex.entryPoint = "main";
	vertex.bufferCount = 1;
	vertex.buffers = &vertexBufferLayout;

	// Fragment state
	WGPUBlendState blend = {};
	blend.color.operation = WGPUBlendOperation_Add;
	blend.color.srcFactor = WGPUBlendFactor_One;
	blend.color.dstFactor = WGPUBlendFactor_Zero;
	blend.alpha.operation = WGPUBlendOperation_Add;
	blend.alpha.srcFactor = WGPUBlendFactor_One;
	blend.alpha.dstFactor = WGPUBlendFactor_Zero;

	WGPUColorTargetState colorTarget = {};
	colorTarget.format = webgpu::getSwapChainFormat(device);
	colorTarget.blend = &blend;
	colorTarget.writeMask = WGPUColorWriteMask_All;

	WGPUFragmentState fragment = {};
	fragment.module = fragMod;
	fragment.entryPoint = "main";
	fragment.targetCount = 1;
	fragment.targets = &colorTarget;


#ifdef __EMSCRIPTEN__
	WGPURenderPipelineDescriptor desc = {};
#else
	WGPURenderPipelineDescriptor desc = {};
#endif
	desc.vertex = vertex;
	desc.fragment = &fragment;
	desc.layout = pipelineLayout;
	// Other states

	// Primitive state
	desc.primitive.frontFace = WGPUFrontFace_CCW;
	desc.primitive.cullMode = WGPUCullMode_None;
	desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
	desc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;

	// Depth Stencil state: You can add depth test in here
	WGPUDepthStencilState* depthSt = new WGPUDepthStencilState();
	depthSt->depthCompare = WGPUCompareFunction_LessEqual;
	depthSt->stencilFront.compare = WGPUCompareFunction_LessEqual;
	depthSt->stencilBack.compare = WGPUCompareFunction_LessEqual;
	depthSt->format = WGPUTextureFormat_Depth32Float;
	depthSt->depthWriteEnabled = true;
	depthSt->depthBias = 0;
	desc.depthStencil = depthSt;

	// Multisample state
	desc.multisample.count = 1;
	desc.multisample.mask = 0xFFFFFFFF;
	desc.multisample.alphaToCoverageEnabled = false;


#ifdef __EMSCRIPTEN__
	pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
#else
	pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);
#endif

	// partial clean-up 
	wgpuPipelineLayoutRelease(pipelineLayout);

	wgpuShaderModuleRelease(fragMod);
	wgpuShaderModuleRelease(vertMod);
}

/**
 * Draws using the above pipeline and buffers.
 */
static bool redraw() {
	
	WGPUTextureView backBufView = wgpuSwapChainGetCurrentTextureView(swapchain);
	WGPURenderPassColorAttachment colorDesc = {};
	colorDesc.view = backBufView;
	colorDesc.loadOp = WGPULoadOp_Clear;
	colorDesc.storeOp = WGPUStoreOp_Store;
	colorDesc.clearColor.r = 0.1f;
	colorDesc.clearColor.g = 0.1f;
	colorDesc.clearColor.b = 0.1f;
	colorDesc.clearColor.a = 0.7f;

	// You can add depth texture in here 
	WGPURenderPassDepthStencilAttachment depthDesc = {};
	depthDesc.view = depthTexView;
	depthDesc.depthLoadOp = WGPULoadOp_Clear;
	depthDesc.depthStoreOp = WGPUStoreOp_Clear;
	depthDesc.clearDepth = 1.0;


	WGPURenderPassDescriptor renderPass = {};
	renderPass.colorAttachmentCount = 1;
	renderPass.colorAttachments = &colorDesc;
	renderPass.depthStencilAttachment = &depthDesc; 

	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);			// create encoder
	WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPass);	// create pass

	wgpuQueueWriteTexture(queue, &texCopy, img, imgWidth * imgHeight * 4, &texDataLayout, &texSize);
	// update the rotation
	rotDeg += 0.5f;
	if (rotDeg >= 360.f)
		rotDeg = 0.0f;

	wgpuQueueWriteBuffer(queue, uRotBuf, 0, &rotDeg, sizeof(rotDeg));
	//wgpuQueueWriteBuffer(queue, uLightBuf, 0, &light1, sizeof(light1));

	// draw the object (comment these five lines to simply clear the screen)
	wgpuRenderPassEncoderSetPipeline(pass, pipeline);
	wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup, 0, 0);
	wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertBuf, 0, 0);
	wgpuRenderPassEncoderSetIndexBuffer(pass, indxBuf, WGPUIndexFormat_Uint16, 0, 0);

	// Instancing checking
	wgpuRenderPassEncoderDrawIndexed(pass, object.faces.size() * 3, 2, 0, 0, 0);

	wgpuRenderPassEncoderEndPass(pass);
	wgpuRenderPassEncoderRelease(pass);														// release pass
	WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);				// create commands
	wgpuCommandEncoderRelease(encoder);														// release encoder

	wgpuQueueSubmit(queue, 1, &commands);
	wgpuCommandBufferRelease(commands);														// release commands


#ifndef __EMSCRIPTEN__
	wgpuSwapChainPresent(swapchain);
#endif
	wgpuTextureViewRelease(backBufView);													// release textureView

	return true;
}

//---------------------------------------

extern "C" int __main__(int /*argc*/, char* /*argv*/[]) {
	//bool phong = true, gouraud = false, toon = false, sphericalTex = false, cylindricalTex = false, bumpmap = false;


	cout << "Choose the example by type the example's number:\n1)\tphong shading with two instance of sphere.\n2)\tphong shading with teapot.\n3)\tgouraud shading with sphere.\n4)\tgouraud shading with teapot.\n5)\ttoon shading with sphere.\n6)\ttoon shading with teapot.\n7)\tspherical texture mapping.\n8)\tcylindrical texture mapping.\n9)\tbumpmap texture mapping.\nnumber of example:\t";
	cin >> example;
	if (example == 1){
		phong = true;
		sphere = true;
	}
	else if (example == 2) {
		phong = true;
		teapot = true;
	}
	else if (example == 3) {
		gouraud = true;
		sphere = true;
	}
	else if (example == 4) {
		gouraud = true;
		teapot = true;
	}
	else if (example == 5) {
		toon = true;
		sphere = true;
	}
	else if (example == 6) {
		toon = true;
		teapot = true;
	}
	else if (example == 7) {
		sphere = true;
		sphericalTex = true;
	}
	else if (example == 8) {
		sphere = true;
		cylindricalTex = true;
	}
	else if (example == 9) {
		sphere = true;
		bumpmap = true;
	}
	if (sphere) {
		object.loadOBJ("../data/sphere.obj");
		translate = glm::translate(glm::vec3(0.0, 0.0, 0.0));
	}
	else if (teapot) {
		object.loadOBJ("../data/teapot.obj");
		translate = glm::translate(glm::vec3(0.217, -1.575, 0.0));
	}
	if (sphericalTex || cylindricalTex) {
		imgWidth = 1440; imgHeight = 720;
		img = stbi_load("../data/earth.jpg", &imgWidth, &imgHeight, &nrChannels, STBI_rgb_alpha);
	}
	else if (bumpmap) {
		imgWidth = 1000; imgHeight = 500;
		img = stbi_load("../data/earthbump.jpg", &imgWidth, &imgHeight, &nrChannels, STBI_rgb_alpha);
	}
	if(phong || teapot)
		cameraPos = glm::vec3(0.0f, 0.0f, -15.0f);
	else if(sphere)
		cameraPos = glm::vec3(0.0f, 0.0f, -2.0f);

	camera = glm::lookAt(cameraPos, target, up);
	model = translate;
	MVP = projection * camera * model;
	MV = camera * model;
	NormalMatrix = glm::transpose(glm::inverse(camera * model));

	//----------- Draw windows and update scene ------------
	if (window::Handle wHnd = window::create(viewWidth, viewHeight, "Hello CS248")) {
		if ((device = webgpu::create(wHnd))) {

			queue = wgpuDeviceGetQueue(device);
			swapchain = webgpu::createSwapChain(device);

			createPipelineAndBuffers();

			window::show(wHnd);
			window::loop(wHnd, redraw);


#ifndef __EMSCRIPTEN__
			wgpuBindGroupRelease(bindGroup);
			wgpuSamplerRelease(samplerTex);
			wgpuSamplerRelease(depthSamplerTex);
			wgpuTextureViewRelease(texView);
			wgpuTextureViewRelease(depthTexView);
			wgpuBufferRelease(indxBuf);
			wgpuBufferRelease(vertBuf);
			wgpuBufferRelease(uNormalBuf);
			wgpuBufferRelease(uLightBuf);
			wgpuBufferRelease(uCameraBuf);
			stbi_image_free(img);

			wgpuRenderPipelineRelease(pipeline);
			wgpuSwapChainRelease(swapchain);
			wgpuQueueRelease(queue);
			wgpuDeviceRelease(device);
#endif
		}
#ifndef __EMSCRIPTEN__
		window::destroy(wHnd);
#endif
	}


	return 0;
}
