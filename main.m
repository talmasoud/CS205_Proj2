function main()
tic
    disp("Nearest Neighbor Feature Selection.");
    data = load('CS205_small_testdata__21.txt');
    
    %Normalizing data except class labe........
    data (:,2:end) = normalize(data(:,2:end));
    
    disp("Type:");
    disp("Enter 1 for Forward Selection");
    disp("Enter 2 for Backward Elimination");
    %choosing the search algorithm based on user's selection
    number = input('Select an option: ');
    if number == 1

        [accuracy, best_feature_set] = forward_selection(data);
            disp(" ");
            disp(['Feature subset [', num2str(best_feature_set), ...
                '] is best at an accuracy of ', num2str(accuracy), '%.']);


    elseif number == 2
        [accuracy, best_feature_set] = backward_elimination(data);
            disp(" ");
            disp(['Feature subset [', num2str(best_feature_set), ...
                '] is best at an accuracy of ', num2str(accuracy), '%.']);
    else
         disp("Incorrect response.");
    end
    toc
end %end of main.....
% Search for the best accurcy and best feature set and return them.
function [best_accuracy, best_feature_set] = forward_selection(data)
best_accuracy = 0;
features = [];
best_feature_set = [];

for i = 1 : size (data,2)-1
    disp(" ");
    disp(['On the ',num2str(i),'th level of the search tree']);
    disp(" ");
    for k = 1 : size (data,2)-1
        if isempty(intersect(features,k))
            features = [features, k];  % adding feature 
            accuracy = leave_one_out_cross_validation(data, features,best_accuracy);
            disp(['Tested feature set (', num2str(features),')  accuracy: ', num2str(accuracy), '%']);
            features(end) = []; % removing feature
            if accuracy > best_accuracy
                best_accuracy = accuracy;
                feature_to_add = k;
                best_feature_set = [features, k];
                disp(['On level ', num2str(i),', feature ', ...
                    num2str(feature_to_add), ' was added to best set!']);
            end
        end
    end
    features = best_feature_set;
end
end
% Search for the best accurcy and best feature set from
% down to top of the tree...
function [overall_best_accuracy, overall_best_feature_set] =backward_elimination(data)
num_of_features = size (data,2)-1;
features = zeros(1, num_of_features);   
overall_best_accuracy = 0;
for i = 1 : num_of_features  
    features(i) = i;
end
accuracy = leave_one_out_cross_validation(data, features,overall_best_accuracy);
disp(['Tested full feature set (', num2str(features), ...
    ') accuracy: ', num2str(accuracy), '%']);
overall_best_accuracy = accuracy;
overall_best_feature_set = features;
while ~isempty(features)
    if size(features,2) > 1
        current_best_accuracy = 0;
        current_best_feature_set = [];
        for i = 1 : size(features,2)
            removed_feature = features(i);
            features(i) = [];   % remove feature from array
            accuracy = leave_one_out_cross_validation(data, features,overall_best_accuracy);
            disp(['Removed feature ', num2str(removed_feature), ...
                ' from feature set (', num2str(features), ...
                ')  accuracy: ', num2str(accuracy), '%']);
            if accuracy > current_best_accuracy
                current_best_accuracy = accuracy;
                current_best_feature_set = features;
                disp(['Feature set [', ...
                    num2str(current_best_feature_set), ...
                    '] is local best set, with accuracy ', ...
                    num2str(current_best_accuracy), '%']);
                if current_best_accuracy > overall_best_accuracy
                    overall_best_accuracy = accuracy;
                    overall_best_feature_set = features;
                    disp(['Feature set [', ...
                        num2str(overall_best_feature_set), ...
                        '] is global best set, with accuracy ', ...
                        num2str(overall_best_accuracy), '%']);
                end
            end
            % add feature back
            features = [features(1:i-1), removed_feature, features(i:end)];
        end
        features = current_best_feature_set;
    else
        removed_feature = features;
        features = [];   % remove feature from array
        accuracy = leave_one_out_cross_validation(data, features,overall_best_accuracy);
        if current_best_accuracy > overall_best_accuracy
            overall_best_accuracy = accuracy;
            overall_best_feature_set = features;
            disp(['Feature set [', ...
                num2str(overall_best_feature_set), ...
                '] is global best set, with accuracy ', ...
                num2str(overall_best_accuracy), '%']);
        end
    end
end
end
function accuracy = leave_one_out_cross_validation(data,features,global_best_acc)
correct_counter = 0;
incorrect_counter = 0;
num_of_members = size(data(:,1));
num_of_members = num_of_members(1,1);
for i = 1 : size(data,1)
    object_to_classify = data(i,2:end); 
    min_distance = Inf;%big number
    for k = 1 : size(data,1)
        if i ~= k % leave-one-out
            new_distance = get_distance(data, features, i, k);
            if new_distance < min_distance
                min_distance = new_distance;
                nearest_member = data(k,1);
            end
        end
    end
    if data(i,1) == nearest_member
        correct_counter = correct_counter + 1;
    end
end
accuracy = (correct_counter / num_of_members) * 100;
end
% Calculating the distance of the first neighbors... 
function distance = get_distance(data, features, origin_point, other_point)
distance = 0;
for i = 1 : numel(features)  % for all features, skipping member column
    curr_feature = features(i);
    distance = distance + ( data(other_point, curr_feature+1) ...
        - data(origin_point, curr_feature+1) )^2;
end
distance = sqrt(distance);
end