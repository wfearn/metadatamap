import os, json

users = {}
for file in os.listdir('../users'):
    if '.DS_Store' in file:
        continue
    # sort the files in the folder and grab only the most recent .txt log
   # logs = []
   # for file in os.listdir('../users/' + user):
   #     if '.txt' in file:
   #         logs.append(file)
   # logs.sort()
   # final = logs[-1]
    
    user = file.replace('.txt','')
    user = user[3:]
    print(user)
    users[user] = {}
    users[user]['iters'] = {}
    with open('../users/' + file) as f:
        # each line contains timestamp, userid, active time, activity, and information regarding the activity
        iters = 0
        totalCorrect = 0
        totalLabeled = 0
        totalAccuracy = 0
        users[user]['iters'][iters] = {}
        for line in f:
            print(line)
            line = line.split('||')
            timeStamp = line[0]
            user = line[1]
            activeTime = int(line[2])
            action = line[3]
            if action == 'INITIAL_LOAD':
                condition = line[5]
                print(users)
                users[user]['startSession'] = timeStamp
                users[user]['inputUncertainty'] = condition
            if action == 'STARTING_TASK':
                users[user]['startTask'] = timeStamp
            if action == 'NEW_DEBATES':
                accuracy = line[4].split(',')
                users[user]['iters'][iters]['startTimeStamp'] = timeStamp
                users[user]['iters'][iters]['startActiveTime'] = activeTime
                users[user]['iters'][iters]['currAccuracy'] = accuracy[1]
                users[user]['iters'][iters]['totalAccuracy'] = accuracy[3]
                totalAccuracy = accuracy[3]
            if action == 'SEND_UPDATE':
                if not line[4]:
                    continue
                labeled = line[3].split(',')
                docs = line[4].split(';')
                correctness = line[5].split(',')
                numCorrect = int(correctness[1])
                # numLabeled = int(labeled[1])
                users[user]['iters'][iters]['updateTimeStamp'] = timeStamp
                users[user]['iters'][iters]['updateActiveTime'] = activeTime
                users[user]['iters'][iters]['activeTimeOnIter'] = users[user]['iters'][iters]['updateActiveTime'] - users[user]['iters'][iters]['startActiveTime']
                users[user]['iters'][iters]['labeled'] = labeled[1]
                users[user]['iters'][iters]['correct'] = numCorrect
                # totalLabeled += numLabeled
                totalCorrect += numCorrect
                iters += 1
                users[user]['iters'][iters] = {}
            users[user]['totalCorrect'] = totalCorrect
            users[user]['totalLabeled'] = totalLabeled
            users[user]['totalPerceivedAccuracy'] = totalAccuracy
            users[user]['numIters'] = iters
        f.close()
print(users)
