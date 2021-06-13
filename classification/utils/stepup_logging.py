# encoding=utf-8
import os,sys
import logging
import logging.handlers #logging不会自动加载子模块?


def setup(program, outdir="logs", use_global=True):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if use_global:
        logger = logging.getLogger() #设置root logger，影响所有logger
    else:
        logger = logging.getLogger(program)  # 设置单个 logger,不影响其他logger
    fmt = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
    logger.setLevel(logging.INFO)

    scr = logging.StreamHandler()
    scr.setFormatter(fmt)

    filename = os.path.join(outdir,program+".log")
    filehandler = logging.handlers.TimedRotatingFileHandler(filename=filename,when='D',backupCount=10,
                                                   encoding='utf-8')

    filehandler.suffix= "%Y-%m-%d.log" #必须设置suffix才可以实现按天删除功能
    filehandler.setFormatter(fmt)
    logger.addHandler(scr)
    logger.addHandler(filehandler)
    return
