#-*- coding: utf-8 -*-


from __future__ import print_function

import os
import numpy as np
np.random.seed(1337)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import *
from gensim.models.word2vec import LineSentence  
from keras.layers import Embedding,Input,TimeDistributed,Dense,Dropout,Bidirectional,LSTM
from keras.utils.np_utils import to_categorical
from keras.models import Model

import codecs

import sys

from __future__ import print_function, unicode_literals
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde
ptrain='./postrain.txt'#正例训练集地址
pval='./posval.txt'#正例验证集地址
ptest='./postest.txt'#正例测试集地址

ntrain='./negtrain.txt'#负例训练集地址
nval='./negval.txt'#负例验证集地址
ntest='./negtest.txt'#负例测试集地址

pathInputPositive_train=ptrain
pathInputPositive_test=ptest
pathInputPositive_val=pval

pathInputNegative_train=ntrain
pathInputNegative_test=ntest
pathInputNegative_val=nval

pathOutputRecord='./Record.txt'#记录结果的文档


Word_Vec_Min_Num = 1
Embedding_Dim = 256
Word_Vec_Window=5


#序列标注的最大序列长度
# MAX_SEQ_LEN=25#按逗号分，每个的长度
MAX_SEQ_LEN=41#不按逗号分，整个长度
#取前几个字符作为训练
# Max_Num_Words = 100000
#LSTM内部投影和输出的维度
LSTM_Dim = 64





   

#将要输入的中文进行id化，建立实际值和id值的对应词典

def getChar_indexIndex_char(pathInputChar, char_index, index_char):
    
    with codecs.open(pathInputChar, 'r', 'utf-8') as f_read:
        for line in f_read.readlines():
            line = line.strip() # 把末尾的'\n'删掉
            
            for subLine in line.split(','):
                chars=subLine.split(" ")

                index = len(char_index)+1
                for char in chars:
                    if(not char in char_index):
                        char_index[char] = index
                        index_char[index]= char
                        index = index+1
                    
    return char_index, index_char
         

char_index={}
index_char={}

char_index, index_char=getChar_indexIndex_char(pathInputPositive_train, char_index, index_char)


char_index, index_char=getChar_indexIndex_char(pathInputPositive_val, char_index, index_char)


char_index, index_char=getChar_indexIndex_char(pathInputNegative_train, char_index, index_char)


char_index, index_char=getChar_indexIndex_char(pathInputNegative_val, char_index, index_char)




num_words = len(char_index)+1#加1是增加了一个EOS

    
EOS = '<EOS>'
indexEOS = len(char_index)+1
index_char[indexEOS] = EOS
char_index[EOS] = indexEOS

#----------------------------------------------------------------------------------

def lcs(a,b):  
    lena=len(a)  
    lenb=len(b)  
    c=[[0 for i in range(lenb+1)] for j in range(lena+1)]  
    flag=[[0 for i in range(lenb+1)] for j in range(lena+1)]  
    for i in range(lena):  
        for j in range(lenb):  
            if a[i]==b[j]:  
                c[i+1][j+1]=c[i][j]+1  
                flag[i+1][j+1]='ok'  
            elif c[i+1][j]>c[i][j+1]:  
                c[i+1][j+1]=c[i+1][j]  
                flag[i+1][j+1]='left'  
            else:  
                c[i+1][j+1]=c[i][j+1]  
                flag[i+1][j+1]='up'  
    return c,flag  

def getListFeatures(seq1, seq2):
    c,flag=lcs(seq1,seq2)

    listFeature1=[]
    preLabel=0
    for j in c[1:]:
        if(j[-1]==preLabel):
            listFeature1.append(1)
        else:
            listFeature1.append(0)
        preLabel=j[-1]

    listFeature2=[]
    preLabel=0
    for j in c[-1][1:]:
        if(j==preLabel):
            listFeature2.append(1)
        else:
            listFeature2.append(0)
        preLabel=j
    return listFeature1, listFeature2

#----------------------------------------------------------------------------------------------------------


def getTrainTest(pathInputChar, label, char_index, x_train_1, x_train_2, attention_train_1, attention_train_2, y_train):
    
    #处理输入序列
    arrayListListChar=[[],[]]
    listAttentions1=[]
    listAttentions2=[]
    listLabel=[]
    
    with codecs.open(pathInputChar, 'r', 'utf-8') as f_read:
        for line in f_read.readlines():
            line = line.strip() # 把末尾的'\n'删掉
            subLines=line.split(",")

            for i in range(len(subLines)):
                subLine=subLines[i]
                listChar=[]
                words=subLine.split(' ')
                lenWords=len(words)
                for j in range(0,lenWords):
                    char=words[j]
                    num=char_index[char]
                    listChar.append(num)
                listChar.append(indexEOS)
                lenWords = len(listChar)
                #不足的补0，因向量得是固定长度
                for j in range(lenWords,MAX_SEQ_LEN):
                    listChar.append(0)
                try:
                    arrayListListChar[i].append(listChar)
                except:
                    print(line)
                    
            listFeature1, listFeature2 = getListFeatures(subLines[0].split(' '), subLines[1].split(' '))
            
            listFeature1.append(1)
            listFeature2.append(1)
            #不足的补0，因向量得是固定长度
            for j in range(len(listFeature1),MAX_SEQ_LEN):
                listFeature1.append(0)
            for j in range(len(listFeature2),MAX_SEQ_LEN):
                listFeature2.append(0)
            listAttentions1.append(listFeature1)
            listAttentions2.append(listFeature2)
                                
            listLabel.append(label)

    x_train_1.extend(arrayListListChar[0])
    x_train_2.extend(arrayListListChar[1])
    attention_train_1.extend(listAttentions1)
    attention_train_2.extend(listAttentions2)
    y_train.extend(listLabel)
    
    return x_train_1, x_train_2, attention_train_1, attention_train_2, y_train

#----------------------------------------------------------------------------------

#训练集的输入序列所对应的网络输入
x_train_1 = []
x_train_2 = []
attention_train_1 = []
attention_train_2 = []
#训练集的输出序列所对应的网络输出
y_train = []


x_train_1, x_train_2, attention_train_1, attention_train_2, y_train = getTrainTest(pathInputNegative_train, 0, char_index, x_train_1, x_train_2, attention_train_1, attention_train_2, y_train)



x_train_1, x_train_2, attention_train_1, attention_train_2, y_train = getTrainTest(pathInputPositive_train, 1, char_index, x_train_1, x_train_2, attention_train_1, attention_train_2, y_train)



#验证集的输入序列所对应的网络输入
x_test_1 = []
x_test_2 = []
attention_test_1 = []
attention_test_2 = []
#验证集的输出序列所对应的网络输出
y_test = []

x_test_1, x_test_2, attention_test_1, attention_test_2, y_test = getTrainTest(pathInputPositive_val, 1, char_index, x_test_1, x_test_2, attention_test_1, attention_test_2, y_test)


x_test_1, x_test_2, attention_test_1, attention_test_2, y_test = getTrainTest(pathInputNegative_val, 0, char_index, x_test_1, x_test_2, attention_test_1, attention_test_2, y_test)


x_train_1 = np.asarray(x_train_1)
x_train_2 = np.asarray(x_train_2)
attention_train_1 = np.asarray(attention_train_1)
attention_train_2 = np.asarray(attention_train_2)
y_train = np.asarray(y_train)
# 转化成one-hot形式，1个数字对应一个词袋向量，如listChar=[0,2]，则temps=[[ 1.  0.  0.][ 0.  0.  1.]]
y_train = to_categorical(y_train,2)

x_test_1 = np.asarray(x_test_1)
x_test_2 = np.asarray(x_test_2)
attention_test_1 = np.asarray(attention_test_1)
attention_test_2 = np.asarray(attention_test_2)
y_test = np.asarray(y_test)
# 转化成one-hot形式，1个数字对应一个词袋向量，如listChar=[0,2]，则temps=[[ 1.  0.  0.][ 0.  0.  1.]]
y_test = to_categorical(y_test,2)

# x_train = np.concatenate((x_train_1, x_train_2), axis=0) 
# x_test = np.concatenate((x_test_1, x_test_2), axis=0)

print(x_train_1.shape)
print(x_train_2.shape)
print(attention_train_1.shape)
print(attention_train_2.shape)
print(y_train.shape)
print(MAX_SEQ_LEN)
print("finish")


"""
Some key layers used for constructing a Capsule Network. These layers can used to construct CapsNet on other dataset, 
not just on MNIST.
*NOTE*: some functions can be implemented in multiple ways, I keep all of them. You can try them for yourself just by
uncommenting them and commenting their counterparts.

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = Conv1D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv1d')(inputs)
    outputs = Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


"""
# The following is another way to implement primary capsule layer. This is much slower.
# Apply Conv2D `n_channels` times and concatenate all capsules
def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    outputs = []
    for _ in range(n_channels):
        output = layers.Conv2D(filters=dim_capsule, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
        outputs.append(layers.Reshape([output.get_shape().as_list()[1] ** 2, dim_capsule])(output))
    outputs = layers.Concatenate(axis=1)(outputs)
    return layers.Lambda(squash)(outputs)
"""


from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class LinerLayer(Layer):
    def __init__(self, **kwargs):
        super(LinerLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        inital_W1= 10.0
        inital_W2= 0.0
        self.W1 = K.variable(inital_W1)
        self.W2 = K.variable(inital_W2)
        self.trainable_weights = [self.W1, self.W2]  
        
        super(LinerLayer,self) 

    def call(self, x, mask=None):
        listResult=x*self.W1+self.W2
        return listResult

    def get_output_shape_for(self, input_shape):
        return input_shape



from keras.models import Sequential
from keras.layers import SimpleRNN, GRU, Activation, Dense, concatenate, dot, add, multiply, Activation, Lambda, Conv1D
from keras.layers.core import Masking, Flatten, Reshape, RepeatVector, Permute, Flatten
from keras.layers.pooling import MaxPooling1D, GlobalMaxPool1D
from keras import backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
bi_GRU = Bidirectional(GRU(LSTM_Dim, unroll=False, return_sequences=True))
# bi_GRU = Conv1D(CNN_filters, 3, strides=1, padding='same', dilation_rate=1, activation='relu')

linerLayer = LinerLayer()
embedding_layer = Embedding(num_words + 1,
                            Embedding_Dim,
#                             weights=[embedding_matrix],
                            input_length=MAX_SEQ_LEN,
                            mask_zero=False, 
                            trainable=True)

input_layer_1 = Input(shape=(MAX_SEQ_LEN,))
embedding_layer_1 = embedding_layer(input_layer_1)
bi_GRU_1 = bi_GRU(embedding_layer_1)
assert bi_GRU.get_output_at(0) == bi_GRU_1
# max_pooling_1 = MaxPooling1D()(bi_GRU_1)
bi_GRU_1= GlobalMaxPool1D()(bi_GRU_1)

input_layer_2 = Input(shape=(MAX_SEQ_LEN,))
embedding_layer_2 = embedding_layer(input_layer_2)
bi_GRU_2 = bi_GRU(embedding_layer_2)
assert bi_GRU.get_output_at(1) == bi_GRU_2
# max_pooling_2 = MaxPooling1D()(bi_GRU_2)
bi_GRU_2 = GlobalMaxPool1D()(bi_GRU_2)
# #----------------求权重------------------------------------------------------
# repeat_GRU_2=RepeatVector(MAX_SEQ_LEN)(bi_GRU_2)
# weight_layer_1=concatenate([bi_GRU_1,repeat_GRU_2],axis=-1)
# weight_layer_1=Activation('tanh')(weight_layer_1)
# weight_layer_1=TimeDistributed(Dense(1))(weight_layer_1)
# weight_layer_1=Lambda(lambda x: K.sum(x, axis=-1))(weight_layer_1)

# #----------------权重归一化操作--------------------------------------------
# # exp_layer =  Lambda(lambda x: K.exp(x))(weight_layer_1)
# onezero_layer_1 = Lambda(lambda x: K.sign(x))(input_layer_1)
# # multiply_layer_1 = multiply([exp_layer, onezero_layer_1])
# # merge_layer_1 = Lambda(lambda x: x/K.sum(x))(multiply_layer_1)
# multiply_layer_1 = multiply([weight_layer_1, onezero_layer_1])
# minus_layer_1 = Lambda(lambda x: (x-1)*1e10)(onezero_layer_1)
# add_layer_1 = add([multiply_layer_1, minus_layer_1])
# merge_layer_1 = Activation('softmax')(add_layer_1)

#-------------------获取权重---------------------------------------------------
attention_layer_1 = Input(shape=(MAX_SEQ_LEN,))
# attention_liner_layer_1 = linerLayer(attention_layer_1)
# assert linerLayer.get_output_at(0) == attention_liner_layer_1
# attention_liner_layer_1 = Activation('softmax')(attention_liner_layer_1)
attention_liner_layer_1 = attention_layer_1
#----------------bi_GRU_1加权平均-------------------------------------
repeat_layer_1 = RepeatVector(2*LSTM_Dim)(attention_liner_layer_1)
repeat_layer_1 = Permute((2, 1))(repeat_layer_1)
dot_layer_1 = multiply([bi_GRU_1,repeat_layer_1])
dot_layer_1 = Lambda(lambda x: K.sum(x, axis=-2, keepdims=True))(dot_layer_1)

#-------------------获取权重---------------------------------------------------
attention_layer_2 = Input(shape=(MAX_SEQ_LEN,))
# attention_liner_layer_2 = linerLayer(attention_layer_2)
# assert linerLayer.get_output_at(1) == attention_liner_layer_2
# attention_liner_layer_2 = Activation('softmax')(attention_liner_layer_2)
attention_liner_layer_2=attention_layer_2
#----------------bi_GRU_2加权平均-------------------------------------
repeat_layer_2 = RepeatVector(2*LSTM_Dim)(attention_liner_layer_2)
repeat_layer_2 = Permute((2, 1))(repeat_layer_2)
dot_layer_2 = multiply([bi_GRU_2,repeat_layer_2])
dot_layer_2 = Lambda(lambda x: K.sum(x, axis=-2, keepdims=True))(dot_layer_2)

#----------------------------------------------------------------------

concat_layer = concatenate([dot_layer_1,dot_layer_2], axis=-2)

primarycaps = Lambda(squash, name='primarycap_squash')(concat_layer)

#-------------------------------------------

# Layer 3: Capsule layer. Routing algorithm works here.
digitcaps = CapsuleLayer(num_capsule=2, dim_capsule=64, routings=2, name='digitcaps')(primarycaps)

# Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
# If using tensorflow, this will not be necessary. :)
out_caps = Length(name='capsnet')(digitcaps)


model = Model(inputs=[input_layer_1, input_layer_2, attention_layer_1, attention_layer_2], outputs=out_caps)
# model.load_weights('./data/hypersno/outputModelWeights/OutputModelWeights_char_Embedding_256_Random_LSTM_256_GRU-Shared-Attention-Given-Trained-GRU1-GRU2-tanh-MLP-Concat-softmax-binary_crossentropy-mask_2017-12-25_17:42:13_epoch_173617.txt')
#打印出模型概况
model.summary()

print("finish model")




def write2file(line):
    with codecs.open(pathOutputRecord, 'a', 'utf-8') as file_write:
        file_write.write(line+"\n")
    

def getListTag(listTagIdOnehot, isOnehot):
    listTag = []
    for i in range(0,len(listTagIdOnehot)):
        indexTag=0
        if(isOnehot):
            indexTag=np.ndarray.argmax(listTagIdOnehot[i])
        else:
            indexTag=listTagIdOnehot[i][0]
        if(indexTag==1):
            listTag.append(str(i)+'_'+str(indexTag))
    return listTag

def getPRF(listListTagIdPred, listListTagIdVal):
    setTagPred=set(getListTag(listListTagIdPred, True))
    setTagVal=set(getListTag(listListTagIdVal, True))
    setRight=setTagPred & setTagVal
    lenRight=len(setRight)
    lenPred=len(setTagPred)
    lenVal=len(setTagVal)
    p=0
    if(lenPred!=0):
        p=lenRight/float(lenPred)
    r=0
    if(lenVal!=0):
        r=lenRight/float(lenVal)
    f=0
    if(p+r!=0):
        f=2*p*r/(p+r)
    return p,r,f,lenRight,lenPred,lenVal

def printPRF(listListTagIdPred, listListTagIdVal):
    p,r,f,lenRight,lenPred,lenVal=getPRF(listListTagIdPred, listListTagIdVal)
    strP=str(round(p*100,2))+"%"
    strR=str(round(r*100,2))+"%"
    strF=str(round(f*100,2))+"%"
    strResult="Precision:"+strP+"\tRecall:"+strR+"\tF-score:"+strF+"\tlenRight:"+str(lenRight)+"\tlenPred:"+str(lenPred)+"\tlenVal:"+str(lenVal)
    print(strResult)
    write2file(strResult)
    return strP,strR,strF
print('finish')

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        1 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

	
import datetime,time
 
from keras.optimizers import Adadelta

adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
model.compile(loss='binary_crossentropy', optimizer=adadelta, metrics=['binary_crossentropy'])


print('Train...')
model.fit([X_words_train, X_pos_train], y_train,
           batch_size=batch_size, nb_epoch=3, callbacks=[conlleval])


numEpoch=120



now = time.strftime("%Y-%m-%d_%H:%M:%S")
pathOutputModelWeights="./outputModelWeights/OutputModelWeights_dsx4602_char_Embedding_256_Random_LSTM_256_GRU-Shared-Attention-Given-Trained-GRU1-GRU2-tanh-MLP-Concat-softmax-binary_crossentropy-mask-EOS_"+now+"_epoch_"+str(i+1)+".txt"
y_pred = model.predict([x_test_1, x_test_2, attention_test_1, attention_test_2],batch_size=128)
PPP,RRR,FFF=printPRF(y_pred, y_test)
model.save_weights(pathOutputModelWeights)
write2file(pathOutputModelWeights)
write2file("")

