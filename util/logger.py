from datetime import datetime


class Log(object):
    @staticmethod
    def d(tag, msg):
        t = str(datetime.now())
        print "%s - %s: %s" % (t, tag, msg)
