src = imread('../../Data/Des/Color Images/7.jpg');
src = double(src)/255 + randn(size(src))*0.1;
c1 = 1.0;
c2 = 1.2;
total_epos = 10;

des = src;
for epos=1:total_epos
    epos
    expection = zeros(size(src));
    for r=2:size(src, 1)-1
        for c=2:size(src, 2)-1
            for h=1:size(src, 3)
                A = c1*(1.0/(1.0+abs(src(r-1,c,h)-src(r,c,h))) * des(r-1,c, h) ...
                    + 1.0/(1.0+abs(src(r+1,c,h)-src(r,c,h))) * des(r+1,c, h) ...
                    + 1.0/(1.0+abs(src(r,c-1,h)-src(r,c,h))) * des(r,c-1, h) ... 
                    + 1.0/(1.0+abs(src(r,c+1,h)-src(r,c,h))) * des(r,c+1, h)) ...
                    + c2 * src(r, c, h);
                expection(r, c, h) = A / sqrt(c1+c2);
            end
        end
    end
    expection = expection ./ sum(sum(sum(expection))) * sum(sum(sum(src)));
    des = expection;
    imshow(des);
end

figure,imshow(src);