function TVV=T_coarse_coarse2(W,s,xX);
Assume_L1=zeros(2*W+1,2*W+1);[M1,N1]=find(Assume_L1==0);
Assume_L2=zeros(s,s);[M2,N2]=find(Assume_L2==0);
for i=1:(2*W+1)^2
    for j=1:(2*W+1)^2
        TVV(i,j)=0;
        for m=1:s^2
            for n=1:s^2
                p1=[(M1(i)-1)*s+M2(m),(N1(i)-1)*s+N2(m)];
                p2=[(M1(j)-1)*s+M2(n),(N1(j)-1)*s+N2(n)];
                TVV(i,j)=TVV(i,j)+myfun2(xX,norm(p1-p2));
            end
        end
    end
end
TVV=TVV/s^4;