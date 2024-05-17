#include "resnet.h"

/**
 * @brief 该函数实现了一个基本块的前向传播
 *
 * @param[in] input 输入特征图的指针
 * @param[in] block ResNet基本块结构体指针
 * @param[out] output:输出特征图指针
 *
 * @return 执行状态，返回 ALGO_NORMAL 表示执行成功，其他值表示执行失败
 */
int resnet_block_forward(Conv2dData* input, ResNetBlock* block, Conv2dData* output)
{
    Conv2dData temp_output; // 创建临时输出特征图
    temp_output.data = (double*)malloc(sizeof(double) * input->row * input->col *
        input->channel); // 为特征图中的data分配对应大小的空间
    if (!temp_output.data) {
        return ALGO_MALLOC_FAIL; // malloc分配失败
    }

    // 第一个卷积层
    int ret = conv2d_bn_no_bias(input, &block->conv1, &temp_output);
    // 结果不正常，直接free释放后返回
    if (ret != ALGO_NORMAL) {
        free(temp_output.data);
        return ret;
    }

    // 激活层
    ret = leaky_relu(0.1, temp_output.data, input->row * input->col * input->channel,
        temp_output.data);
    // 结果不正常，直接free释放后返回
    if (ret != ALGO_NORMAL) {
        free(temp_output.data);
        return ret;
    }

    // 第二个卷积层
    ret = conv2d_bn_no_bias(&temp_output, &block->conv2, output);
    // 结果不正常，直接free释放后返回
    if (ret != ALGO_NORMAL) {
        free(temp_output.data);
        return ret;
    }

    // 快捷连接，残差操作，将第一个卷积层之前的输入与第二个卷积层后的输出相连
    for (uint16_t i = 0; i < input->row * input->col * input->channel; ++i) {
        output->data[i] += input->data[i];
    }

    // 正常释放内存
    free(temp_output.data);
    return ALGO_NORMAL;
}

/**
 * @brief 在给定输入数据和 ResNet 网络结构的情况下，计算前向传播结果
 *
 * @param[in] input 输入特征图数据
 * @param[in] resnet ResNet 网络结构
 * @param[out] output 输出数据的指针，用于存储前向传播结果
 *
 * @return 执行状态，返回 ALGO_NORMAL 表示执行成功，其他值表示执行失败
 */
int resnet_forward(Conv2dData* input, ResNet* resnet, double* output)
{
    // 创建临时输出特征图和块输出特征图
    uint16_t conv_out_len = 0;
    Conv2dData conv_out;
    memset(&conv_out, 0, sizeof(Conv2dData));
    conv_out_len = cal_conv_out_len(input->col, 0, 2, 2);
    conv_out.data = (double*)malloc(sizeof(double) * conv_out_len * 2);
    if (!conv_out.data) {
        return ALGO_MALLOC_FAIL;
    }

    Conv2dData block_output;
    memset(&block_output, 0, sizeof(Conv2dData));
    block_output.data = (double*)malloc(sizeof(double) * input->row * input->col * input->channel);
    if (!block_output.data) {
        return ALGO_MALLOC_FAIL;
    }

    // 初始卷积层
    int ret = conv2d_bn_no_bias(input, &resnet->initial_conv, &conv_out);
    if (ret != ALGO_NORMAL) {
        free(block_output.data);
        return ret;
    }

    // 激活层
    ret = leaky_relu(0.1, conv_out.data, input->row * input->col * input->channel,
        conv_out.data);
    if (ret != ALGO_NORMAL) {
        free(block_output.data);
        return ret;
    }

    // 通过resnet块前向传播实现resnet网络的传播
    for (uint16_t i = 0; i < resnet->num_blocks; ++i) {
        ret = resnet_block_forward(input, &resnet->blocks[i], &block_output);
        if (ret != ALGO_NORMAL) {
            free(block_output.data);
            return ret;
        }
        conv_out = block_output;
    }

    // 线性层
    ret = linear_layer(conv_out.data, &resnet->linear, output);

    // 释放内存
    free(block_output.data);
    return ret;
}

/**
 * @brief 初始化Conv2dFilter卷积核参数
 *
 * @param[in] filter 输入Conv2dFilter结构体指针
 *
 * @return 空
 */
void initialize_filter(Conv2dFilter* filter)
{
    filter->channel = 1;
    filter->col = 2;
    filter->row = 1;
    filter->filter_num = 2;
    filter->data = model_0_weight;
}

/**
 * @brief 初始化BN参数
 *
 * @param[in] bn 输入BatchNorm2d结构体指针
 *
 * @return 空
 */
void initialize_bn(BatchNorm2d* bn)
{
    bn->size = 2;
    bn->beta = model_1_bias;
    bn->gamma = model_1_weight;
    bn->mean = model_1_running_mean;
    bn->var = model_1_running_var;
}

/**
 * @brief 初始化ResNetBlock
 *
 * @param[in] bn 输入ResNetBlock结构体指针
 *
 * @return 空
 */
void initialize_ResNetBlock(ResNetBlock* rb)
{
    // 分配初始内存
    rb->conv1.pad = 0;
    rb->conv1.stride = 2;
    initialize_filter(&rb->conv1.filter);
    initialize_bn(&rb->conv1.bn);

    rb->conv2.pad = 0;
    rb->conv2.stride = 2;
    initialize_filter(&rb->conv2.filter);
    initialize_bn(&rb->conv2.bn);
}

/**
 * @brief 初始化线性层配置
 *
 * @param[in] bn 输入LinearParam结构体指针
 *
 * @return 空
 */
void initialize_LinearParam(LinearParam* lp)
{
    lp->inp_size = 240;
    lp->fea_size = 2;
    lp->weight = output_weight;
    lp->bias = output_bias;
}

/**
 * @brief 初始化resnet网络结构体
 *
 * @param[in] resnet 输入resnet结构体指针
 *
 * @return 空
 */
void initialize_resnet(ResNet* resnet)
{
    resnet->initial_conv.filter.channel = 1;
    resnet->initial_conv.filter.col = 2;
    resnet->initial_conv.filter.row = 1;
    resnet->initial_conv.filter.filter_num = 2;
    resnet->initial_conv.filter.data = model_0_weight;

    resnet->initial_conv.bn.beta = model_1_bias;
    resnet->initial_conv.bn.gamma = model_1_weight;
    resnet->initial_conv.bn.mean = model_1_running_mean;
    resnet->initial_conv.bn.var = model_1_running_var;
    resnet->initial_conv.bn.size = 2;

    // 初始化卷积层配置
    resnet->initial_conv.pad = 0;
    resnet->initial_conv.stride = 2;

    // 初始化卷积核参数
    initialize_filter(&resnet->initial_conv.filter);

    // 初始化BN参数
    initialize_bn(&resnet->initial_conv.bn);

    // 初始化ResNet块数组
    for (int i = 0; i < res_blocks_num; ++i) {
        initialize_ResNetBlock(&resnet->blocks[i]);
    }

    // 初始化线性层配置
    initialize_LinearParam(&resnet->linear);
}

/**
 * @brief 释放卷积核资源
 *
 * @param[in] filter 输入Conv2dFilter结构体指针
 *
 * @return 空
 */
void release_filter(Conv2dFilter* filter)
{
    if (filter->data) {
        free(filter->data);
    }
}

/**
 * @brief 释放bn资源
 *
 * @param[in] bn 输入bn结构体指针
 *
 * @return 空
 */
void release_bn(BatchNorm2d* bn)
{
    if (bn->beta) {
        free(bn->beta); // 释放 beta 参数内存
        bn->beta = NULL; // 将指针置为 NULL，避免误用
    }
    if (bn->gamma) {
        free(bn->gamma); // 释放 gamma 参数内存
        bn->gamma = NULL; // 将指针置为 NULL，避免误用
    }
    if (bn->mean) {
        free(bn->mean); // 释放 mean 参数内存
        bn->mean = NULL; // 将指针置为 NULL，避免误用
    }
    if (bn->var) {
        free(bn->var); // 释放 var 参数内存
        bn->var = NULL; // 将指针置为 NULL，避免误用
    }
}

/**
 * @brief 释放resnet_block资源
 *
 * @param[in] block 输入ResNetBlock结构体指针
 *
 * @return 空
 */
void release_resnet_block(ResNetBlock* block)
{
    // 释放第一个卷积层的卷积核参数内存
    release_filter(&block->conv1.filter);
    // 释放第一个卷积层的 BN 参数内存
    release_bn(&block->conv1.bn);

    // 释放第二个卷积层的卷积核参数内存
    release_filter(&block->conv2.filter);
    // 释放第二个卷积层的 BN 参数内存
    release_bn(&block->conv2.bn);
}

/**
 * @brief 释放resnet资源
 *
 * @param[in] resnet 输入resnet结构体指针
 *
 * @return 空
 */
void release_resnet(ResNet* resnet)
{
    // 释放初始卷积层的卷积核内存
    release_filter(&resnet->initial_conv.filter);

    // 释放初始卷积层的bn内存
    release_bn(&resnet->initial_conv.bn);

    // 释放resnet块数组中每个块的内存
    for (int i = 0; i < res_blocks_num; ++i) {
        release_resnet_block(&resnet->blocks[i]);
    }

    // 释放resnet块数组内存
    free(resnet->blocks);

    // 释放线性层的权重和偏置内存
    free(&resnet->linear.weight);
    free(&resnet->linear.bias);

}