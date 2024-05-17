#pragma once

#include "conv.h"
#include "model_parameters.h"

static res_blocks_num = 10;

/**
 * @brief ResNet��ṹ�壬��ʾResNet�е�һ��������
 *
 * �ýṹ������������������
 */
typedef struct ResNetBlock {
    Conv2dConfig conv1; // ��һ������������
    Conv2dConfig conv2; // �ڶ�������������
} ResNetBlock;

/**
 * @brief ResNet����ṹ�壬��ʾ��������
 *
 * �ýṹ�����ResNet����ĳ�ʼ����������á��������ResNet����������Բ�����
 */
typedef struct _ResNet {
    Conv2dConfig initial_conv; // ��ʼ�����������ýṹ��
    ResNetBlock blocks[10]; // һ�����飬�洢��� ResNet ������
    uint16_t num_blocks; // �����������
    LinearParam linear; // �������Բ�����
} ResNet;

/**
 * @brief �ú���ʵ����һ���������ǰ�򴫲�
 *
 * @param
 * input:��������ͼ��ָ��
 * block:ResNet������ṹ��ָ��
 * output:�������ͼָ��
 *
 * @return ִ��״̬������ ALGO_NORMAL ��ʾִ�гɹ�������ֵ��ʾִ��ʧ��
 */
int resnet_block_forward(Conv2dData* input, ResNetBlock* block, Conv2dData* output);

/**
 * @brief �ڸ����������ݺ� ResNet ����ṹ������£�����ǰ�򴫲����
 *
 * @param[in] input ��������ͼ����
 * @param[in] resnet ResNet ����ṹ
 * @param[out] output ������ݵ�ָ�룬���ڴ洢ǰ�򴫲����
 *
 * @return ִ��״̬������ ALGO_NORMAL ��ʾִ�гɹ�������ֵ��ʾִ��ʧ��
 */
int resnet_forward(Conv2dData* input, ResNet* resnet, double* output);

/**
 * @brief ��ʼ��Conv2dFilter����˲���
 *
 * @param[in] filter ����Conv2dFilter�ṹ��ָ��
 *
 * @return ��
 */
void initialize_filter(Conv2dFilter* filter);

/**
 * @brief ��ʼ��BN����
 *
 * @param[in] bn ����BatchNorm2d�ṹ��ָ��
 *
 * @return ��
 */
void initialize_bn(BatchNorm2d* bn);

/**
 * @brief ��ʼ��ResNetBlock
 *
 * @param[in] bn ����ResNetBlock�ṹ��ָ��
 *
 * @return ��
 */
void initialize_ResNetBlock(ResNetBlock* rb);

/**
 * @brief ��ʼ�����Բ�����
 *
 * @param[in] bn ����LinearParam�ṹ��ָ��
 *
 * @return ��
 */
void initialize_LinearParam(LinearParam* lp);

/**
 * @brief ��ʼ��resnet����ṹ��
 *
 * @param[in] resnet ����resnet�ṹ��ָ��
 *
 * @return ��
 */
void initialize_resnet(ResNet* resnet);

/**
 * @brief �ͷž������Դ
 *
 * @param[in] filter ����Conv2dFilter�ṹ��ָ��
 *
 * @return ��
 */
void release_filter(Conv2dFilter* filter);

/**
 * @brief �ͷ�bn��Դ
 *
 * @param[in] bn ����bn�ṹ��ָ��
 *
 * @return ��
 */
void release_bn(BatchNorm2d* bn);

/**
 * @brief �ͷ�resnet_block��Դ
 *
 * @param[in] block ����ResNetBlock�ṹ��ָ��
 *
 * @return ��
 */
void release_resnet_block(ResNetBlock* block);

/**
 * @brief �ͷ�resnet��Դ
 *
 * @param[in] resnet ����resnet�ṹ��ָ��
 *
 * @return ��
 */
void release_resnet(ResNet* resnet);