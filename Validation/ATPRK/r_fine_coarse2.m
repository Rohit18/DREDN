function rvV=r_fine_coarse2(p_vm,W,s,xX);
Assume_L1=zeros(2*W+1,2*W+1);[M1,N1]=find(Assume_L1==0);
Assume_L2=zeros(s,s);[M2,N2]=find(Assume_L2==0);
for i=1:(2*W+1)^2
    rvV(i,1)=0;
    for m=1:s^2
        p1=[(M1(i)-1)*s+M2(m),(N1(i)-1)*s+N2(m)];
        rvV(i,1)=rvV(i,1)+myfun2(xX,norm(p_vm-p1));
    end
end
rvV=rvV/s^2;