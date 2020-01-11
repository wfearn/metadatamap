import json
import wikipedia
import wptools

tweet_file = 'tweets.json'
user_file = 'users.json'
new_user_file = 'users_final.json'

congresspeople = list()

with open(new_user_file, 'r') as f:
    for line in f:
        json_line = json.loads(line.encode('utf-8'))
        congresspeople.append(json_line['name'])


with open(user_file, 'r') as f, open(new_user_file, 'a') as g:
    for line in f:
        json_line = json.loads(line.encode('utf-8'))
        name = json_line['name']

        if name in congresspeople:
            print(f'Skipping {name}')
            continue
        else:
            print(f'Don\'t have {name}')

        #person = wikipedia.page(name)
        #print('Wikipedia page:', person.categories)
        try:
            so = wptools.page(name).get_parse()
            party = so.data['infobox']['party'].lower()
            if 'republican' in party:
                party = 'Republican'
            elif 'democrat' in party:
                party = 'Democratic'
            else:
                party = 'Independent'

            json_line['party'] = party
            new_line = json.dumps(json_line)
            g.write(f'{new_line}\n')

        except (AttributeError, LookupError, TypeError) as e:
            party = input(f'Which political party does {name} belong to? ')

            if party == 'r':
                party = 'Republican'
            elif party == 'd':
                party = 'Democratic'
            elif party == 'i':
                party = 'Independent'
            else:
                party = input('Incorrect Input, try again: ')

            json_line['party'] = party
            new_line = json.dumps(json_line)
            g.write(f'{new_line}\n')
