function T = find_target(x, D)

[c, ~] = size(D); % # of prototypes by feature dim.
distance = zeros(c,1);

for i = 1:c
    y = D(i,:);
    distance(i) = norm(x - y);
end

[~,id] = min(distance);
T = D(id,:);