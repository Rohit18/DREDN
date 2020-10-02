function Z=ATPK_DS(Coarse,s,Sill_min,Range_min,L_sill,L_range,rate,H);
[a1,b1]=size(Coarse);
W=2;
Coarse_extend1=[repmat(Coarse(:,1),[1,W]),Coarse,repmat(Coarse(:,end),[1,W])];%%%extend columns
Coarse_extend=[repmat(Coarse_extend1(1,:),[W,1]);Coarse_extend1;repmat(Coarse_extend1(end,:),[W,1])];%%%extend rows

x0=[0.1,1];%%%%%x0 is the initial value for fitting
for h=1:H
    rh(h)=semivariogram(Coarse,h);
end
[xa1,resnorm]=lsqcurvefit(@myfun2,x0,s:s:s*H,rh);Fa1=myfun2(xa1,1:1:s*H);
xp_best=ATP_deconvolution0(H,s,xa1,Sill_min,Range_min,L_sill,L_range,rate);Fp=myfun2(xp_best,[1:1:s*H]);
raa0=r_area_area2(H,s,xp_best);raa=raa0(2:H+1,1)-raa0(1,1);[xa2,resnorm]=lsqcurvefit(@myfun2,x0,s:s:s*H,raa');Fa2=myfun2(xa2,[1:1:s*H]);
xp_best_matrix=xp_best;

yita1=ATPK_noinform_yita_new(s,W,xp_best);
P_vm=ATPK_noinform_new(s,W,Coarse_extend,yita1);
Z=P_vm(W*s+1:end-W*s,W*s+1:end-W*s);
