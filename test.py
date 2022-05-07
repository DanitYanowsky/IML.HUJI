def solution(blocks):
    array=[0] * len(blocks)
    for index,num in enumerate(blocks):
        if index>0 and num>=blocks[index-1]:
            array[index] =  array[index-1] +1
    for index in range(len(blocks)-1,-1,-1):
        if index>0 and num>=blocks[index-1]:
            array[index] =  array[index] +1
    return max(array)+1
print(solution([1,2,1,2,1]))