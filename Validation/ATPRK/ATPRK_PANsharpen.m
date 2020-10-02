function [xrc1,RB,Z]=ATPRK_PANsharpen(Coarse,PAN,Sill_min,Range_min,L_sill,L_range,rate,H);
[a1,b1]=size(Coarse);
[a2,b2]=size(PAN);
s=a2/a1;

%%%%correlation analysis
J_PAN_upscaled=dowmsample_plane(PAN,s);
x=0:0.0001:1;y=x;
GB0=reshape(J_PAN_upscaled,1,a1*b1);
GB1=reshape(Coarse,1,a1*b1);

%%%%%linear regression modeling
x0=[0.1,1];%%%%%x0 is the initial value for fitting
[xrc1,resnorm]=lsqcurvefit(@myfun1,x0,GB0,GB1);Ff1=myfun1(xrc1,PAN);Z_R=reshape(Ff1,a1*s,b1*s);

%%%%%residual calculation
Z_R_upscaled=dowmsample_plane(Z_R,s);RB=Coarse-Z_R_upscaled;

%%%%%ATPK for residuals, Deconvolution is achieved by a trail-and-error procedure
Z_ATPK=ATPK_DS(RB,s,Sill_min,Range_min,L_sill,L_range,rate,H);
Z=Z_R+Z_ATPK;
