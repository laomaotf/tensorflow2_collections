# encoding=utf-8
import os,sys
import logging
import logging.handlers #logging不会自动加载子模块?

class EZLOG:
    def __init__(self,program, outdir="logs"):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.logger = logging.getLogger(program)
        fmt = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
        self.logger.setLevel(logging.INFO)

        scr = logging.StreamHandler()
        scr.setFormatter(fmt)

        filename = os.path.join(outdir,program+".log")
        filehandler = logging.handlers.TimedRotatingFileHandler(filename=filename,when='D',backupCount=10,
                                                       encoding='utf-8')

        filehandler.suffix= "%Y-%m-%d.log" #必须设置suffix才可以实现按天删除功能
        filehandler.setFormatter(fmt)
        self.logger.addHandler(scr)
        self.logger.addHandler(filehandler)
        return

    def info(self,msg):
        self.logger.info(msg)
    def warning(self,msg):
        self.logger.warning(msg)
    def error(self,msg):
        self.logger.error(msg)

