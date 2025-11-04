#include <iostream>
#include <vector>
#include <algorithm>

struct Item {
    int value;
    int weight;
    double val_per_weight;
};

void give_val_per_weight(std::vector<Item>& items)
{
    for (auto& item : items)
        item.val_per_weight = item.value / item.weight;
}

bool compareItems(Item i1, Item i2) {
   return i1.val_per_weight > i2.val_per_weight;
}

double fractionalKnapsack(int capacity, std::vector<Item>& items) {

    give_val_per_weight(items);

    sort(items.begin(), items.end(), compareItems);

    double totalValue = 0.0;
    int currentWeight = 0;

    for (const auto& item : items)
    {
        if (currentWeight < capacity)
        {
            if (currentWeight + item.weight <= capacity)
            {
                totalValue += item.value;
                currentWeight += item.weight;
            }
            else
            {
                double remaining_weight = capacity - currentWeight;
                totalValue += item.val_per_weight * remaining_weight;
                currentWeight = capacity;
            }
        }
    }

    return totalValue;
}

int main() {
    int capacity = 50;
    std::vector<Item> items = {
        {60, 10}, {100, 20}, {120, 30}
    };

    double maxValue = fractionalKnapsack(capacity, items);
    std::cout << "Maximum value in knapsack: " << maxValue << std::endl;
    
    return 0;
}