%% Question 1-4
p = [5 9 17 17 5 125];
q = [9 5 9 121 1 1];
for i= 1:numel(q)
    figure(i)
    fftwave(p(i),q(i),128)
end


%% Question 5
x= reshape(1:16,4,4)
y=fftshift(x)