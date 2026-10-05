
TOTAL_DIST = 3250

class Tile:
    NAME = ['EmptySpace', 'Object', 'Enemy', 'Mario']
    EMPTY_SPACE = 0
    OBJECT = 1
    ENEMY = 2
    MARIO = 3

    @staticmethod
    def name(index):
        return Tile.NAME[index]

class Window:
    Width = 16
    Height = 13

    @staticmethod
    def getFrameSize():
        return Window.Width * Window.Height

class Action:
    NAME = ['Up', 'Left', 'Down', 'Right', 'A', 'B']
    NO_ACTION = 'NO_ACTION'

    @staticmethod
    def index(name):
        return Action.NAME.index(name)

    @staticmethod
    def set(action, name):
        action[Action.index(name)] = 1

    @staticmethod
    def unset(action, name):
        action[Action.index(name)] = 0

    @staticmethod
    def act(names):
        action = Action.empty() 
        if type(names) is list: # a tuple of names
            for n in names:
                if n == Action.NO_ACTION:
                    continue
                action[Action.index(n)] = 1
        elif names != Action.NO_ACTION: # a single names
            action[Action.index(names)] = 1
        return action

    @staticmethod
    def empty():
        return len(Action.NAME) * [0]

    @staticmethod
    def names(action):
        l = []
        for i, act in enumerate(action):
            if act == 1:
                l.append(Action.NAME[i])
        if len(l)==0:
            l.append(Action.NO_ACTION)
        return l

class Time:
    TOTAL_GAME_TIME = 400
