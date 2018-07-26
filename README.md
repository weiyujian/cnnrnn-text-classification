# cnnrnn-text-classification

#cnnrnn text classification of tensorflow

#This code belongs to the "Implementing a CNN for Text Classification in Tensorflow" blog post

#I modified it to deal with Chinese text classification and add a cnn-rnn model in it.

#And I add a rnn model with attention, rnn-cnn model, cnn-rnn model and cnn concat rnn model to compare different model effect.

#The experiment shows that cnn concat rnn model did best and then cnn model.

#语料来源：使用THUCNews的一个子集进行训练与测试，训练集大小：5w，开发集5k，测试集5k。

#这个子集可以在此下载：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

#experiment result:
#cnn模型: 开发集：915, 验证集：0.899
#rnn模型:开发集：0.918, 验证集：0.895
#cnnrnn模型：开发集：0.932， 验证集:0.898
#rnncnn模型:开发集：0.904, 验证集：0.874
#rnn concat cnn模型： 开发集：0.932， 验证集：0.906
