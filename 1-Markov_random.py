
import numpy as np
import random as rm


# 状态空间
states = ["Sleep","Run","Icecream"]
 
# 可能的事件序列
transitionName = [["Sleep","Run","Icecream"],["Sleep","Run","Icecream"],["Sleep","Run","Icecream"]]
 
# 概率矩阵（转移矩阵）
transitionMatrix = [[0.2,0.6,0.2],[0.1,0.6,0.3],[0.2,0.7,0.1]]

#check
if sum(transitionMatrix[0])==1 and sum(transitionMatrix[1])== 1 and sum(transitionMatrix[1]) ==1:
    print("All is gonna be okay, you should move on!! ;)")
else: print("Somewhere, something went wrong. Transition matrix, perhaps?")

# 实现了可以预测状态的马尔可夫模型的函数。
def activity_forecast(activityToday,days):

    #print("Start state: " + activityToday)
    # 应该记录选择的状态序列。这里现在只有初始状态。
    activityList = [activityToday]
    i = 0
    # prob用于计算 activityList 的概率
    prob = 1

    #匹配当前状态在表中属于哪个
    state_index = states.index(activityToday)

    while i < days:
        change = np.random.choice(transitionName[state_index],replace=True,p=transitionMatrix[state_index])
        #change activityToday 内部用index来索引更好。
        activity_index = transitionName[state_index].index(change)
        prob *= transitionMatrix[state_index][activity_index]
        
        activityToday = change
        activityList.append(activityToday)
        #响应的索引
        #activityList.append(transitionMatrix[state_index][activity_index])
        i += 1
        state_index = states.index(activityToday)
    '''
    print("Possible states: " + str(activityList))
    print("End state after "+ str(days) + " days: " + activityToday)
    print("Probability of the possible sequence of states: " + str(prob))
    '''
    return  activityList

#output
print("transitionMatrix is:")
print(transitionMatrix[0],"\n",transitionMatrix[1],"\n",transitionMatrix[2])
print("transitionName is:")
print(transitionName[0],"\n",transitionName[1],"\n",transitionName[2],"\n")

random = 0

if random == 1:
# 随机预测 X  天后的可能状态
    # 选择初始状态
    activityToday = "Sleep"
    days = 3
    activityList = activity_forecast(activityToday,days)

else:
    #以某种状况结束的概率
    list_activity = []
    count = 0
    activityToday = "Sleep"
    EndActivity = "Run"
    days = 2
    for iterations in range(1,10000):
        list_activity.append(activity_forecast(activityToday,days))
    for smaller_list in list_activity:
        if smaller_list[2] == EndActivity:
            count += 1
    percentage = (count/10000) * 100
    print("The probability of starting at state:"+ str(activityToday) + " and ending at state:" + str(EndActivity) + " = " + str(percentage) + "%")

    