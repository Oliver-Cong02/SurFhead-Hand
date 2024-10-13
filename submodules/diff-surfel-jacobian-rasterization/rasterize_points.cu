/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

#define CHECK_INPUT(x)											\               
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

const int MAX_GAUSSIANS_PER_PIXEL = 30;  // Set this value as needed

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

// std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,torch::Tensor>
// std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& Jacobians,
	const torch::Tensor& Jacobians_inv,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const bool tight_pruning_threshold)
{// from torch
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(colors);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(Jacobians);
  CHECK_INPUT(Jacobians_inv);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);
//   auto bool_opts = means3D.options().dtype(torch::kBool); //? means3D is already on CUDA

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_others = torch::full({3+3+1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

//   torch::Tensor gaussian_indices = torch::full({H, W, MAX_GAUSSIANS_PER_PIXEL}, -1, int_opts);  //? Initialize with -1
//   torch::Tensor n_contrib_pixel = torch::full({H, W}, 0, int_opts); //?
//   torch::Tensor n_contrib_pixel = torch::full({H, W}, 0, int_opts).to(torch::kCUDA);
//   torch::Tensor visible_points = torch::full({P}, false, means3D.options().dtype(torch::kBool));


  torch::Tensor n_contrib_pixel = torch::full({H, W}, 0, int_opts); //.to(torch::kCUDA); // GPU memory
  torch::Tensor visible_points = torch::full({P}, false, means3D.options().dtype(torch::kBool));//.to(torch::kCUDA); // GPU memory

	
  torch::Tensor top_weights = torch::full({H, W, MAX_GAUSSIANS_PER_PIXEL}, -1.0, float_opts);
  torch::Tensor top_depths = torch::full({H, W, MAX_GAUSSIANS_PER_PIXEL}, -1.0, float_opts); 
   
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		Jacobians.contiguous().data_ptr<float>(),
		Jacobians_inv.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_others.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		// gaussian_indices.contiguous().data<int>(),  //? Pass the new tensor
        n_contrib_pixel.contiguous().data<int>(), //? Pass the new tensor
		MAX_GAUSSIANS_PER_PIXEL,
		top_weights.contiguous().data<float>(),
		top_depths.contiguous().data<float>(),
		visible_points.contiguous().data<bool>(),
		tight_pruning_threshold,
		debug);
  }
//   return std::make_tuple(rendered, out_color, out_others, radii, geomBuffer, binningBuffer, imgBuffer);
// }
	// return std::make_tuple(rendered, out_color, out_others, radii, gaussian_indices, n_contrib_pixel, top_weights, top_depths, visible_points, geomBuffer, binningBuffer, imgBuffer);
	// return std::make_tuple(rendered, out_color, out_others, radii, n_contrib_pixel, visible_points, geomBuffer, binningBuffer, imgBuffer);
	return std::make_tuple(rendered, out_color, out_others, radii, n_contrib_pixel, visible_points, top_weights, top_depths, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& Jacobians,
	const torch::Tensor& Jacobians_inv,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(colors);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(Jacobians);
  CHECK_INPUT(Jacobians_inv);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);
  CHECK_INPUT(binningBuffer);
  CHECK_INPUT(imageBuffer);
  CHECK_INPUT(geomBuffer);

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor abs_dL_dmean2D = torch::zeros({P, 2}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  Jacobians.data_ptr<float>(), //? which is jacobian
	  Jacobians_inv.data_ptr<float>(),
	  transMat_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_others.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dnormal.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dtransMat.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  abs_dL_dmean2D.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}