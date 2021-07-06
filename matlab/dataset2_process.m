xset = zeros(1800,64,64);
label=[];
for i = 1:300
     str_t = num2str(i);%将数字转换为图像文件名
     str=strcat('D:\NEU-CLS\Cr_',str_t);
     str = strcat(str,'.bmp');%连接图像文件名和后缀名.bmp便于下一步自动读取图像
     im = imread(str,'bmp');%读取图像
     im = imresize(im,[64 64]);
    xset(i,:,:) = im;% 存入矩阵
    imwrite(im,['D:\NLS_data\cr\',str_t,'.jpg']);
    label=[label,1];
end
for i = 301:600
     str_t = num2str(i-300);%将数字转换为图像文件名
     str=strcat('D:\NEU-CLS\In_',str_t);
     str = strcat(str,'.bmp');%连接图像文件名和后缀名.bmp便于下一步自动读取图像
     im = imread(str,'bmp');%读取图像
     im = imresize(im,[64 64]);
    xset(i,:,:) = im;% 存入矩阵
    imwrite(im,['D:\NLS_data\in\',str_t,'.jpg']);
    label=[label,2];
end
for i = 601:900
     str_t = num2str(i-600);%将数字转换为图像文件名
     str=strcat('D:\NEU-CLS\Pa_',str_t);
     str = strcat(str,'.bmp');%连接图像文件名和后缀名.bmp便于下一步自动读取图像
     im = imread(str,'bmp');%读取图像
     im = imresize(im,[64 64]);
    xset(i,:,:) = im;% 存入矩阵
    imwrite(im,['D:\NLS_data\pa\',str_t,'.jpg']);
    label=[label,3];
end
for i = 901:1200
     str_t = num2str(i-900);%将数字转换为图像文件名
     str=strcat('D:\NEU-CLS\PS_',str_t);
     str = strcat(str,'.bmp');%连接图像文件名和后缀名.bmp便于下一步自动读取图像
     im = imread(str,'bmp');%读取图像
     im = imresize(im,[64 64]);
    xset(i,:,:) = im;% 存入矩阵
    imwrite(im,['D:\NLS_data\ps\',str_t,'.jpg']);
    label=[label,4];
end
for i = 1201:1500
     str_t = num2str(i-1200);%将数字转换为图像文件名
     str=strcat('D:\NEU-CLS\RS_',str_t);
     str = strcat(str,'.bmp');%连接图像文件名和后缀名.bmp便于下一步自动读取图像
     im = imread(str,'bmp');%读取图像
     im = imresize(im,[64 64]);
    xset(i,:,:) = im;% 存入矩阵
    imwrite(im,['D:\NLS_data\rs\',str_t,'.jpg']);
    label=[label,5];
end
for i = 1501:1800
     str_t = num2str(i-1500);%将数字转换为图像文件名
     str=strcat('D:\NEU-CLS\Sc_',str_t);
     str = strcat(str,'.bmp');%连接图像文件名和后缀名.bmp便于下一步自动读取图像
     im = imread(str,'bmp');%读取图像
     im = imresize(im,[64 64]);
    xset(i,:,:) = im;% 存入矩阵
    imwrite(im,['D:\NLS_data\sc\',str_t,'.jpg']);
    label=[label,6];
end
save('D:/Data2_resized.mat', 'xset', 'label'); 