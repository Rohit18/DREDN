function S=dowmsample_plane(plane,s);
[sizec,sized]=size(plane);
S=zeros(sizec/s,sized/s);
for i=1:s:sizec
    for j=1:s:sized
        m=(i+s-1)/s; n=(j+s-1)/s;
        for p=i:i+s-1
            for q=j:j+s-1
                S(m,n)=S(m,n)+plane(p,q); 
            end
        end
    end
end
S=S/s^2;