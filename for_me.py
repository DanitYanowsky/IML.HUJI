from itertools import count
import heapq



def bubble_sort(nums):
        for index_out,num_out in enumerate(nums):
            for i_in, num_in in enumerate(nums[0:len(nums)-index_out]):
                counter= len(nums)-index_out
                if i_in<(len(nums)-index_out)-1 and num_in> nums[i_in+1]:
                    temp = nums[i_in+1]
                    nums[i_in+1] = num_in
                    nums[i_in] = temp
                else:
                    counter -=1
            if counter ==0:
                break
        return nums
    
def selection_sort(nums):
    for index_out,num_out in enumerate(nums):
        temp = nums[index_out]
        index = index_out
        for i_in, num_in in enumerate(nums[index_out:]):
            if i_in<(len(nums)) and num_in< temp:
                temp = num_in
                index = i_in
        nums[index] = nums[index_out]
        nums[index_out] = temp
    return nums
a= [2,3,4]
# print(a[:-1])
# print(selection_sort([3,2,5,1]))

def insertion_sort(nums):
    for j in range(len(nums)):
        pointer = j
        for index in range(len(nums[:j+1])-1,-1,-1):
            if nums[pointer]<nums[index]:
                temp = nums[index]
                nums[index]=nums[pointer]
                nums[pointer] =temp 
                pointer = index
    return nums

def heapsort(nums):
    heapq.heapify(nums)
    array = []
    while nums:
        array.append(heapq.heappop(nums))
    return array

def counting_sort(nums, k):
    array = [0]*k
    for index, val in enumerate(nums):
        array[val-1]+=1
    index = 0
    for j, val in enumerate(array):
        if array[j]==0:
            continue
        while array[j]!=0:
            nums[index] = j+1
            array[j]-=1
            index+=1
    return nums

def radix_sort(nums,d):
    j=0
    while d>0:
        j+=1
        def key(x):
            return x%(10**j)
        nums.sort(key=key)
        d-=1
    return nums

class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """

        
        m = len(grid)
        n = len(grid[0])
        sum=0
        def dfs(index_tuple):
            if index_tuple[0]>m or index_tuple[1]>n:
                return
            if grid[index_tuple[0]][index_tuple[1]]=="1":
                grid[index_tuple[0]][index_tuple[1]]="0"
                for i in range(index_tuple[0]+1,m):
                    if grid[i][index_tuple[1]]!="1":
                        break
                    dfs((i,index_tuple[1]))
                for i in range(index_tuple[0]-1,-1,-1):
                    if grid[i][index_tuple[1]]!="1":
                        break
                    dfs((i,index_tuple[1]))
                for j in range(index_tuple[1]+1,n):
                    if grid[index_tuple[0]][j]!="1":
                        break
                    dfs((index_tuple[0],j))
                for j in range(index_tuple[1]-1,-1,-1):
                    if grid[index_tuple[0]][j]!="1":
                        break
                    dfs((index_tuple[0],j))
        for i in range(m):
            for j in range(n):
                if grid[i][j]=="1":
                    dfs((i,j))
                    sum+=1
        return sum
a=Solution()
# print(a.numIslands([["1","1","1"],["0","1","0"],["1","1","1"]]))

# print(radix_sort([211,32,53,22],3))
class Solution(object):
    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        def merge(A,B):
            # merges 2 sorted array into 1 sortet array

            C = [0]*(len(A) + len(B))
            n = len(C)
            c, a, b = 0, 0, 0
            while a < len(A) and b < len(B):
                if A[a] < B[b]:
                    C[c] = A[a]
                    a += 1
                else:
                    C[c] = B[b]
                    b += 1
                c += 1
            if a < len(A):  # B is done
                 while a < len(A):
                     C[c] = A[a]
                     a += 1
                     c += 1

            if b < len(B):  # B is done
                while b < len(B):
                    C[c] = B[b]
                    b += 1
                    c += 1
            return C

        def merge_sort_recursive(A):
            n = len(A)
            if n <= 1:
                return A
            R = merge_sort_recursive(A[:n//2])
            L = merge_sort_recursive(A[n//2:])
            return merge(L, R)        
        return merge_sort_recursive(nums)
a=Solution()
a.sortArray([5,2,3,1])
      
                    