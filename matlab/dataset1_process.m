clear; clc
pathset1='F:\ZYG\SIGS\dataset1\train';
cd(pathset1)
nlist=dir();
nimg=zeros(length(nlist)-2,1);
for i=1:length(nlist)-2
    fname=nlist(i+2).name;
    plist=dir(fname);
    nimg(i)=length(plist)-2;
end
ntotal=sum(nimg);
xset=zeros([64,64,ntotal],'uint8');
idlist=zeros(10,1);
for i=1:9
    idlist(i+1)=idlist(i)+nimg(i);
end
label=zeros(ntotal,1);
for i=1:9
    label(idlist(i)+1:idlist(i+1))=i;
end
for i=1:length(nlist)-2
    fname=nlist(i+2).name;
    plist=dir(fname);
    for j=1:length(plist)-2
        iname=[fname,'\',plist(j+2).name];
        img=imread(iname);
        xset(:,:,j+idlist(i))=img(:,:,1);
    end
end
save('trainset1.mat','xset','label','idlist','-v7.3')
% imshow(reshape(xset(5054,:,:),64,64))



