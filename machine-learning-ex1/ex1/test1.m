v = rand(7, 1);
w = rand(7, 1);

z = 0;
for i = 1:7
  z = z + v(i) * w(i);
end
initial = z

d = w * v'