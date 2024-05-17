#include "resnet.h"

/**
 * @brief �ú���ʵ����һ���������ǰ�򴫲�
 *
 * @param[in] input ��������ͼ��ָ��
 * @param[in] block ResNet������ṹ��ָ��
 * @param[out] output:�������ͼָ��
 *
 * @return ִ��״̬������ ALGO_NORMAL ��ʾִ�гɹ�������ֵ��ʾִ��ʧ��
 */
int resnet_block_forward(Conv2dData* input, ResNetBlock* block, Conv2dData* output)
{
    Conv2dData temp_output; // ������ʱ�������ͼ
    temp_output.data = (double*)malloc(sizeof(double) * input->row * input->col *
        input->channel); // Ϊ����ͼ�е�data�����Ӧ��С�Ŀռ�
    if (!temp_output.data) {
        return ALGO_MALLOC_FAIL; // malloc����ʧ��
    }

    // ��һ�������
    int ret = conv2d_bn_no_bias(input, &block->conv1, &temp_output);
    // �����������ֱ��free�ͷź󷵻�
    if (ret != ALGO_NORMAL) {
        free(temp_output.data);
        return ret;
    }

    // �����
    ret = leaky_relu(0.1, temp_output.data, input->row * input->col * input->channel,
        temp_output.data);
    // �����������ֱ��free�ͷź󷵻�
    if (ret != ALGO_NORMAL) {
        free(temp_output.data);
        return ret;
    }

    // �ڶ��������
    ret = conv2d_bn_no_bias(&temp_output, &block->conv2, output);
    // �����������ֱ��free�ͷź󷵻�
    if (ret != ALGO_NORMAL) {
        free(temp_output.data);
        return ret;
    }

    // ������ӣ��в����������һ�������֮ǰ��������ڶ�����������������
    for (uint16_t i = 0; i < input->row * input->col * input->channel; ++i) {
        output->data[i] += input->data[i];
    }

    // �����ͷ��ڴ�
    free(temp_output.data);
    return ALGO_NORMAL;
}

/**
 * @brief �ڸ����������ݺ� ResNet ����ṹ������£�����ǰ�򴫲����
 *
 * @param[in] input ��������ͼ����
 * @param[in] resnet ResNet ����ṹ
 * @param[out] output ������ݵ�ָ�룬���ڴ洢ǰ�򴫲����
 *
 * @return ִ��״̬������ ALGO_NORMAL ��ʾִ�гɹ�������ֵ��ʾִ��ʧ��
 */
int resnet_forward(Conv2dData* input, ResNet* resnet, double* output)
{
    // ������ʱ�������ͼ�Ϳ��������ͼ
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

    // ��ʼ�����
    int ret = conv2d_bn_no_bias(input, &resnet->initial_conv, &conv_out);
    if (ret != ALGO_NORMAL) {
        free(block_output.data);
        return ret;
    }

    // �����
    ret = leaky_relu(0.1, conv_out.data, input->row * input->col * input->channel,
        conv_out.data);
    if (ret != ALGO_NORMAL) {
        free(block_output.data);
        return ret;
    }

    // ͨ��resnet��ǰ�򴫲�ʵ��resnet����Ĵ���
    for (uint16_t i = 0; i < resnet->num_blocks; ++i) {
        ret = resnet_block_forward(input, &resnet->blocks[i], &block_output);
        if (ret != ALGO_NORMAL) {
            free(block_output.data);
            return ret;
        }
        conv_out = block_output;
    }

    // ���Բ�
    ret = linear_layer(conv_out.data, &resnet->linear, output);

    // �ͷ��ڴ�
    free(block_output.data);
    return ret;
}

/**
 * @brief ��ʼ��Conv2dFilter����˲���
 *
 * @param[in] filter ����Conv2dFilter�ṹ��ָ��
 *
 * @return ��
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
 * @brief ��ʼ��BN����
 *
 * @param[in] bn ����BatchNorm2d�ṹ��ָ��
 *
 * @return ��
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
 * @brief ��ʼ��ResNetBlock
 *
 * @param[in] bn ����ResNetBlock�ṹ��ָ��
 *
 * @return ��
 */
void initialize_ResNetBlock(ResNetBlock* rb)
{
    // �����ʼ�ڴ�
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
 * @brief ��ʼ�����Բ�����
 *
 * @param[in] bn ����LinearParam�ṹ��ָ��
 *
 * @return ��
 */
void initialize_LinearParam(LinearParam* lp)
{
    lp->inp_size = 240;
    lp->fea_size = 2;
    lp->weight = output_weight;
    lp->bias = output_bias;
}

/**
 * @brief ��ʼ��resnet����ṹ��
 *
 * @param[in] resnet ����resnet�ṹ��ָ��
 *
 * @return ��
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

    // ��ʼ�����������
    resnet->initial_conv.pad = 0;
    resnet->initial_conv.stride = 2;

    // ��ʼ������˲���
    initialize_filter(&resnet->initial_conv.filter);

    // ��ʼ��BN����
    initialize_bn(&resnet->initial_conv.bn);

    // ��ʼ��ResNet������
    for (int i = 0; i < res_blocks_num; ++i) {
        initialize_ResNetBlock(&resnet->blocks[i]);
    }

    // ��ʼ�����Բ�����
    initialize_LinearParam(&resnet->linear);
}

/**
 * @brief �ͷž������Դ
 *
 * @param[in] filter ����Conv2dFilter�ṹ��ָ��
 *
 * @return ��
 */
void release_filter(Conv2dFilter* filter)
{
    if (filter->data) {
        free(filter->data);
    }
}

/**
 * @brief �ͷ�bn��Դ
 *
 * @param[in] bn ����bn�ṹ��ָ��
 *
 * @return ��
 */
void release_bn(BatchNorm2d* bn)
{
    if (bn->beta) {
        free(bn->beta); // �ͷ� beta �����ڴ�
        bn->beta = NULL; // ��ָ����Ϊ NULL����������
    }
    if (bn->gamma) {
        free(bn->gamma); // �ͷ� gamma �����ڴ�
        bn->gamma = NULL; // ��ָ����Ϊ NULL����������
    }
    if (bn->mean) {
        free(bn->mean); // �ͷ� mean �����ڴ�
        bn->mean = NULL; // ��ָ����Ϊ NULL����������
    }
    if (bn->var) {
        free(bn->var); // �ͷ� var �����ڴ�
        bn->var = NULL; // ��ָ����Ϊ NULL����������
    }
}

/**
 * @brief �ͷ�resnet_block��Դ
 *
 * @param[in] block ����ResNetBlock�ṹ��ָ��
 *
 * @return ��
 */
void release_resnet_block(ResNetBlock* block)
{
    // �ͷŵ�һ�������ľ���˲����ڴ�
    release_filter(&block->conv1.filter);
    // �ͷŵ�һ�������� BN �����ڴ�
    release_bn(&block->conv1.bn);

    // �ͷŵڶ��������ľ���˲����ڴ�
    release_filter(&block->conv2.filter);
    // �ͷŵڶ��������� BN �����ڴ�
    release_bn(&block->conv2.bn);
}

/**
 * @brief �ͷ�resnet��Դ
 *
 * @param[in] resnet ����resnet�ṹ��ָ��
 *
 * @return ��
 */
void release_resnet(ResNet* resnet)
{
    // �ͷų�ʼ�����ľ�����ڴ�
    release_filter(&resnet->initial_conv.filter);

    // �ͷų�ʼ������bn�ڴ�
    release_bn(&resnet->initial_conv.bn);

    // �ͷ�resnet��������ÿ������ڴ�
    for (int i = 0; i < res_blocks_num; ++i) {
        release_resnet_block(&resnet->blocks[i]);
    }

    // �ͷ�resnet�������ڴ�
    free(resnet->blocks);

    // �ͷ����Բ��Ȩ�غ�ƫ���ڴ�
    free(&resnet->linear.weight);
    free(&resnet->linear.bias);

}