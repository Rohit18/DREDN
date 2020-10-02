%%%%This is the code for ATPRK produced by Dr Qunming Wang; Email: wqm11111@126.com
%%%%Copyright belong to Qunming Wang
%%%%When using the code, please cite the fowllowing papers
%%%%Q. Wang, W. Shi, P. M. Atkinson, Y. Zhao. Downscaling MODIS images with area-to-point regression kriging. Remote Sensing of Environment, 2015, 166: 191¨C204.
%%%%Q. Wang, Y. Zhang, A. Onojeghuo, X. Zhu, P. M. Atkinson. Enhancing spatio-temporal fusion of MODIS and Landsat data by incorporating 250 m MODIS data. IEEE JSTARS 2017, 10(9): 4116¨C4123

%clear all;
%load MODIS_500m;%%500m MODIS data in a image cube (4 bands, exluding band 5)
%load MODIS_250m;%%250m MODIS data in a image cube (2 bands)
s=4;
%I_MS=MODIS_500m;
%I_PAN=MODIS_250m;

I_MS = im2double(im);
I_PAN = im2double(im_pan);

%%%%correlation analysis
I_PAN_upscaled1=dowmsample_cube(I_PAN,s);%%%upscaling the 250 m data to 500m
for i=1:4
    for j=1:2
        [RMSE0,CC0]=evaluate_relation(I_MS(:,:,i),I_PAN_upscaled1(:,:,j));
        CC_matrix(i,j)=CC0;
    end
end
[II,JJ]=max(CC_matrix,[],2);

Sill_min=1;
Range_min=0.5;
L_sill=20;
L_range=20;
rate=0.1;
H=20;

tic
for i=1:4
    [xrc1,RB0,Z0]=ATPRK_PANsharpen(I_MS(:,:,i),I_PAN(:,:,JJ(i)),Sill_min,Range_min,L_sill,L_range,rate,H);
    %%%%you can also consider using more than one fine band for sharpening,
    %%%%with function ATPRK_MSsharpen
    %Z(:,:,i)=Z0;%%%Choice for original ATPRK
    Z(:,:,i)=xrc1(1)+xrc1(2)*I_PAN(:,:,JJ(i))+imresize(RB0,s,'bicubic');%%%Choice for approximate ATPRK, as presented in the JSTARS paper
end
alltime=toc
