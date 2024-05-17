#pragma once

#include "conv.h"
#include "model_parameters.h"

static res_blocks_num = 10;

/**
 * @brief ResNet块结构体，表示ResNet中的一个基本块
 *
 * 该结构体包含两个卷积层配置
 */
typedef struct ResNetBlock {
    Conv2dConfig conv1; // 第一个卷积层的配置
    Conv2dConfig conv2; // 第二个卷积层的配置
} ResNetBlock;

/**
 * @brief ResNet网络结构体，表示整个网络
 *
 * 该结构体包含ResNet网络的初始化卷积层配置、多个基本ResNet块和最后的线性层配置
 */
typedef struct _ResNet {
    Conv2dConfig initial_conv; // 初始化卷积层的配置结构体
    ResNetBlock blocks[10]; // 一个数组，存储多个 ResNet 基本块
    uint16_t num_blocks; // 基本块的数量
    LinearParam linear; // 最后的线性层配置
} ResNet;

/**
 * @brief 该函数实现了一个基本块的前向传播
 *
 * @param
 * input:输入特征图的指针
 * block:ResNet基本块结构体指针
 * output:输出特征图指针
 *
 * @return 执行状态，返回 ALGO_NORMAL 表示执行成功，其他值表示执行失败
 */
int resnet_block_forward(Conv2dData* input, ResNetBlock* block, Conv2dData* output);

/**
 * @brief 在给定输入数据和 ResNet 网络结构的情况下，计算前向传播结果
 *
 * @param[in] input 输入特征图数据
 * @param[in] resnet ResNet 网络结构
 * @param[out] output 输出数据的指针，用于存储前向传播结果
 *
 * @return 执行状态，返回 ALGO_NORMAL 表示执行成功，其他值表示执行失败
 */
int resnet_forward(Conv2dData* input, ResNet* resnet, double* output);

/**
 * @brief 初始化Conv2dFilter卷积核参数
 *
 * @param[in] filter 输入Conv2dFilter结构体指针
 *
 * @return 空
 */
void initialize_filter(Conv2dFilter* filter);

/**
 * @brief 初始化BN参数
 *
 * @param[in] bn 输入BatchNorm2d结构体指针
 *
 * @return 空
 */
void initialize_bn(BatchNorm2d* bn);

/**
 * @brief 初始化ResNetBlock
 *
 * @param[in] bn 输入ResNetBlock结构体指针
 *
 * @return 空
 */
void initialize_ResNetBlock(ResNetBlock* rb);

/**
 * @brief 初始化线性层配置
 *
 * @param[in] bn 输入LinearParam结构体指针
 *
 * @return 空
 */
void initialize_LinearParam(LinearParam* lp);

/**
 * @brief 初始化resnet网络结构体
 *
 * @param[in] resnet 输入resnet结构体指针
 *
 * @return 空
 */
void initialize_resnet(ResNet* resnet);

/**
 * @brief 释放卷积核资源
 *
 * @param[in] filter 输入Conv2dFilter结构体指针
 *
 * @return 空
 */
void release_filter(Conv2dFilter* filter);

/**
 * @brief 释放bn资源
 *
 * @param[in] bn 输入bn结构体指针
 *
 * @return 空
 */
void release_bn(BatchNorm2d* bn);

/**
 * @brief 释放resnet_block资源
 *
 * @param[in] block 输入ResNetBlock结构体指针
 *
 * @return 空
 */
void release_resnet_block(ResNetBlock* block);

/**
 * @brief 释放resnet资源
 *
 * @param[in] resnet 输入resnet结构体指针
 *
 * @return 空
 */
void release_resnet(ResNet* resnet);