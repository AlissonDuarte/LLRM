from math import sqrt

#those functions have the goal to create relationships over users preferences
def eucliedean(base, user1, user2):
    si = {}
    for item in base[user1]:
        if item in base[user2]:
            si[item] = 1
            
    if len(si) == 0:
        return 0
    
    data_sum = sum(
        [pow(base[user1][item] - base[user2][item], 2) 
            for item in base[user1] 
                if item in base[user2]]
    )
    return 1/(1+sqrt(data_sum))


def getSimilarity(base, user):
    total = {}
    simi_sum = {}
    for item in base:
        if item == user:
            continue
    
        sim = eucliedean(base, user, item)
        
        if sim <= 0:
            continue
        
        for data in base[item]:
            if data not in base[user] or base[user][data] == 0:
                total.setdefault(data, 0)
                total[data] +=base[item][data] * sim
                simi_sum.setdefault(data, 0)
                simi_sum[item] += sim
                
    ranking  = [(sub_total/simi_sum[item], item) for item, sub_total in total.items()]
    ranking.sort()
    ranking.reverse()
    return ranking[0:30]