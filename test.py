import sys

while True:
    sys.stdout.write('>> ')
    input_string = input()
    commands = input_string.split(' ')

    for token in commands:
        sys.stdout.write(token)

    if input_string == 'exit':
        exit(0)