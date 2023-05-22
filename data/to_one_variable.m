% save .mat with added labels
list    = dir(fullfile('', '*.mat'));
nFile   = length(list);
X = 0; Y = 0; Z = 0; Labels = 0;
for i = 1:nFile
    clear x y z labels
    load(list(i).name);
    X = [X; x];
    Y = [Y; y];
    Z = [Z; z];
    Labels = [Labels; labels];
end
X(1) = [];
Y(1) = [];
Z(1) = [];
Labels(1) = [];

%delete labels==0 and labels ==5;
delete_0 = find(Labels == 0);
X(delete_0) = [];
Y(delete_0) = [];
Z(delete_0) = [];
Labels(delete_0) = [];

delete_5 = find(Labels == 5);
X(delete_5) = [];
Y(delete_5) = [];
Z(delete_5) = [];
Labels(delete_5) = [];

save('all_dogs.mat','X','Y','Z','Labels')