function S=dowmsample_cube(cube,s);
[sizec,sized]=size(cube(:,:,1));
[Nb,rc]=size(D3_D2(cube));
S=zeros(sizec/s,sized/s,Nb);
for i=1:s:sizec
    for j=1:s:sized
        m=(i+s-1)/s; n=(j+s-1)/s;
        for p=i:i+s-1
            for q=j:j+s-1
                S(m,n,:)=S(m,n,:)+cube(p,q,:); 
            end
        end
    end
end
S=S/s^2;