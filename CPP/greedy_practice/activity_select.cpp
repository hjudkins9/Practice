#include <iostream>
#include <vector>
#include <algorithm>

struct Activity {
    int start;
    int finish;
};

// You will need a comparison function for sorting
bool compareActivities(Activity a1, Activity a2) {
    // TODO: Implement the sorting logic
    return a1.finish < a2.finish; // Example: sort by finish time
}

int selectMaxActivities(std::vector<Activity>& activities) {
    // 1. Sort the activities based on your greedy strategy
    std::sort(activities.begin(), activities.end(), compareActivities);

    int count = 0;
    // 2. Keep track of the finish time of the last selected activity
    int lastFinishTime = 0; 

    // 3. Iterate through the sorted activities
    for (const auto& activity : activities) {
        
        if (activity.start >= lastFinishTime) 
        { 
            count ++;
            lastFinishTime = activity.finish;
        }
    }

    return count;
}

int main() {
    // Define your vector of activities
    std::vector<Activity> activities = { 
        {1, 4}, {3, 5}, {0, 6},
        {5, 7}, {3, 9}, {5, 9},
        {6, 10}, {8, 11}, {8, 12},
        {2, 14}, {12, 16}

    };
    
    // Call your function and print the result
    std:: cout << "Max activities: " << selectMaxActivities(activities) << std::endl;
    
    return 0;
}