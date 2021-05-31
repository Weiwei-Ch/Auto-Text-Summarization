# Auto-Text-Summarization

文本摘要学习笔记和代码

本主题里内容为本人在学习自动文本摘要过程中做的笔记以及跑的基于transformer模型代码

1.模型：基于transformer block的算法，代码可参见transformer.py、self_attention_encoder和self_attention_decoder.py文件；

2.注意点：a.用到了Pointer-Generator Network，保证了输出内容来自输入内容，代码见decoding.py;

         b.用的是lr schedules方法（动态lr);
         
3.运用结果见‘模型结果截图’图片。

为了上传方便，把一些文件夹删除了，直接上传了py文件，如果有想跑此项目的朋友，可以认真看下代码文件，改下import文件夹等；
