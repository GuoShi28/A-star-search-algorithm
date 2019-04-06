# A star search algorithm

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

A homework of algorithm in HIT 

## Question description
Utilizing A* algorithm to find the path from start point (red) to target point (cross) with the lowest cost. And need to satisfy following requests and rules:

* gray blocks are barriers and unaccessable.
* the cost of moving four straight directions, i.e., up, down, left and right is 1 and the cost of moving four diagonal directions is sqrt(2).
* Walking though some special terrains need to cost extra cost. Orange blocks represent desert and the terrian cost is 4. Blue blocks represent river and the terrian cost is 2. White blocks are normal terrian and the terrian cost is 0.
* Total cost euqal to the sum of moving cost and terrian cost.

## Test
[Demo] `main.py`

## Results
* Input Terrian Map

![](https://github.com/GuoShi28/A-star-search-algorithm/blob/master/images/input_map.png)  


* Result Path

![](https://github.com/GuoShi28/A-star-search-algorithm/blob/master/images/output_map.png) 

## Requirements and Dependencies

* Python 3
* OpenCV
