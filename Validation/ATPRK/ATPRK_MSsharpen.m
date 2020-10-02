function [xrc1,RB,Z]=ATPRK_MSsharpen(Coarse,PAN,Sill_min,Range_min,L_sill,L_range,rate,H,w,PSF);
%%%Different from ATPRK_PANsharpen, PAN here should be a image cube 
%%%that is, multiple fine resolution bands, rather than a single fine band in ATPRK_PANsharpen.
[a1,b1]=size(Coarse);
[a2,b2,c2]=size(PAN);
s=a2/a1;

%%%%correlation analysis
J_PAN_upscaled=dowmsample_cube(PAN,s,w,PSF);
GB0=D3_D2(J_PAN_upscaled);
GB1=reshape(Coarse,1,a1*b1);

%%%%%linear regression modeling
x0=[0.1,1];%%%%%x0 is the initial value for fitting
%[xrc1,resnorm]=lsqcurvefit(@myfun3,x0,GB0,GB1);
xrc1=lsqlin([ones(a1*b1,1),GB0'],GB1,[],[]) ;
GBF=D3_D2(PAN);
Ff1=[ones(a2*b2,1),GBF']*xrc1;Z_R=reshape(Ff1,a1*s,b1*s);

%%%%%residual calculation
Z_R_upscaled=dowmsample_plane(Z_R,s,w,PSF);
RB=Coarse-Z_R_upscaled;

%%%%%ATPK for residuals, Deconvolution is achieved by a trail-and-error procedure
Z_ATPK=ATPK_DS(RB,s,Sill_min,Range_min,L_sill,L_range,rate,H);
Z=Z_R+Z_ATPK;