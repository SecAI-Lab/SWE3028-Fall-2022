"""
hint + nohint model(mine)
"""
import datetime
import glob
import sys
import os
import time
import argparse
from skimage import color
import warnings
import cv2


from PyQt6.QtGui import QColor, QPen, QImage, QPainter, QIcon
from PyQt6.QtWidgets import QCheckBox, QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget, QApplication, QFileDialog
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QPoint, QPointF
from timm.models import create_model
import torch

# import modeling

import segmentation_models_pytorch as smp
import torch.nn as nn

################################
import math
from functools import partial

import numpy as np
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import drop_path, to_2tuple
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from signal import signal, SIGPIPE, SIG_DFL


class GUIGamut(QWidget):
    update_color = pyqtSignal(object)

    def __init__(self, gamut_size=110):
        QWidget.__init__(self)
        self.gamut_size = gamut_size
        self.win_size = gamut_size * 2  # divided by 4
        self.setFixedSize(self.win_size, self.win_size)
        self.ab_grid = abGrid(gamut_size=gamut_size, D=1)
        self.reset()

    def set_gamut(self, l_in=50):
        self.l_in = l_in
        self.ab_map, self.mask = self.ab_grid.update_gamut(l_in=l_in)
        self.update()

    def set_ab(self, color):
        self.color = color
        self.lab = rgb2lab_1d(self.color)
        x, y = self.ab_grid.ab2xy(self.lab[1], self.lab[2])
        self.pos = QPointF(x, y)
        self.update()

    def is_valid_point(self, pos):
        if pos is None or self.mask is None:
            return False
        else:
            x = pos.x()
            y = pos.y()
            if x >= 0 and y >= 0 and x < self.win_size and y < self.win_size:
                return self.mask[y, x]
            else:
                return False

    def update_ui(self, pos):
        self.pos = pos
        a, b = self.ab_grid.xy2ab(pos.x(), pos.y())
        # get color we need L
        L = self.l_in
        lab = np.array([L, a, b])
        color = lab2rgb_1d(lab, clip=True, dtype='uint8')
        # self.emit(SIGNAL('update_color'), color)
        self.update_color.emit(color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(event.rect(), Qt.GlobalColor.white)
        if self.ab_map is not None:
            ab_map = cv2.resize(self.ab_map, (self.win_size, self.win_size))
            qImg = QImage(ab_map.tostring(), self.win_size, self.win_size, QImage.Format.Format_RGB888)
            painter.drawImage(0, 0, qImg)

        painter.setPen(QPen(Qt.GlobalColor.gray, 3, Qt.PenStyle.DotLine, cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(self.win_size // 2, 0, self.win_size // 2, self.win_size)
        painter.drawLine(0, self.win_size // 2, self.win_size, self.win_size // 2)
        if self.pos is not None:
            painter.setPen(QPen(Qt.GlobalColor.black, 2, Qt.SolidLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
            w = 5
            x = self.pos.x()
            y = self.pos.y()
            painter.drawLine(x - w, y, x + w, y)
            painter.drawLine(x, y - w, x, y + w)
        painter.end()

    def mousePressEvent(self, event):
        pos = event.pos()

        if event.button() == Qt.LeftButton and self.is_valid_point(pos):  # click the point
            self.update_ui(pos)
            self.mouseClicked = True

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self.is_valid_point(pos):
            if self.mouseClicked:
                self.update_ui(pos)

    def mouseReleaseEvent(self, event):
        self.mouseClicked = False

    def sizeHint(self):
        return QSize(self.win_size, self.win_size)

    def reset(self):
        self.ab_map = None
        self.mask = None
        self.color = None
        self.lab = None
        self.pos = None
        self.mouseClicked = False
        self.update()


global mode
mode = "ours"


def mode_info(md):
    global mode
    mode = md
    print(f"colorization mode selected: {mode}")
    return mode


class GUIDraw(QWidget):
    # Signals
    update_color = pyqtSignal(str)
    update_gammut = pyqtSignal(object)
    used_colors = pyqtSignal(object)
    update_ab = pyqtSignal(object)
    update_result = pyqtSignal(object)

    def __init__(self, model=None, nohint_model=None, load_size=224, win_size=512, device='cpu'):
        QWidget.__init__(self)
        self.image_file = None
        self.pos = None
        self.model = model
        # add
        self.nohint_model = nohint_model

        self.win_size = win_size
        self.load_size = load_size
        self.device = device
        self.setFixedSize(win_size, win_size)
        self.uiControl = UIControl(win_size=win_size, load_size=load_size)
        self.move(win_size, win_size)
        self.movie = True
        self.init_color()  # initialize color
        self.im_gray3 = None
        self.eraseMode = False
        self.ui_mode = 'none'  # stroke or point
        self.image_loaded = False
        self.use_gray = True
        self.total_images = 0
        self.image_id = 0

    def clock_count(self):
        self.count_secs -= 1
        self.update()

    def init_result(self, image_file):
        # self.read_image(image_file.encode('utf-8'))  # read an image
        self.read_image(image_file)  # read an image
        ##############################
        # my model
        im_full = cv2.resize(self.im_full, (768, 768), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(im_full, cv2.COLOR_BGR2GRAY)
        gray = np.stack([gray, gray, gray], -1)
        l_img = cv2.cvtColor(gray, cv2.COLOR_BGR2LAB)[:, :, [0]].transpose((2, 0, 1))
        l_img = torch.from_numpy(l_img).type(torch.FloatTensor).to(self.device) / 255
        ab = self.nohint_model(l_img.unsqueeze(0))[0]  # .detach().cpu().numpy().transpose((1,2,0))

        lab = torch.cat([l_img, ab], axis=0).permute(1, 2, 0).cpu().detach().numpy() * 255  # h,w,c
        lab = lab.astype(np.uint8)
        self.my_results = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # 왜.. gbr밖에

        #######
        # 저장용
        ab = ab.permute(1, 2, 0).cpu().detach().numpy() * 255
        ab = cv2.resize(ab, (self.im_full.shape[1], self.im_full.shape[0]), interpolation=cv2.INTER_AREA)  # INTER_CUBIC
        im_l = cv2.cvtColor(self.im_full, cv2.COLOR_BGR2LAB)[:, :, [0]]
        org_my_results = np.concatenate([im_l, ab], axis=2)
        org_my_results = org_my_results.astype(np.uint8)

        self.org_my_results = cv2.cvtColor(org_my_results, cv2.COLOR_LAB2BGR)  # 왜.. gbr밖에

        ##############################
        #
        self.reset()

    def get_batches(self, img_dir):
        self.img_list = glob.glob(os.path.join(img_dir, '*.JPEG'))
        self.total_images = len(self.img_list)
        img_first = self.img_list[0]
        self.init_result(img_first)

    def nextImage(self):
        self.save_result()
        self.image_id += 1
        if self.image_id == self.total_images:
            print('you have finished all the results')
            sys.exit()
        img_current = self.img_list[self.image_id]
        # self.reset()
        self.init_result(img_current)
        self.reset_timer()

    def read_image(self, image_file):
        # self.result = None
        self.image_loaded = True
        self.image_file = image_file
        print(image_file)
        im_bgr = cv2.imread(image_file)
        self.im_full = im_bgr.copy()
        # get image for display
        h, w, c = self.im_full.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        self.scale = float(self.win_size) / self.load_size
        print('scale = %f' % self.scale)
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)

        self.im_win = cv2.resize(self.im_full, (rw, rh), interpolation=cv2.INTER_AREA)  # INTER_CUBIC

        self.dw = int((self.win_size - rw) // 2)
        self.dh = int((self.win_size - rh) // 2)
        self.win_w = rw
        self.win_h = rh
        self.uiControl.setImageSize((rw, rh))
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        self.gray_win = cv2.resize(self.im_gray3, (rw, rh), interpolation=cv2.INTER_AREA)  # INTER_CUBIC
        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_AREA)  # INTER_CUBIC
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lab_win = color.rgb2lab(self.im_win[:, :, ::-1])

        self.org_im_l = color.rgb2lab(self.im_full[:, :, ::-1])[:, :, [0]]
        self.im_lab = color.rgb2lab(im_bgr[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.l_win = lab_win[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

    def update_im(self):
        self.update()
        QApplication.processEvents()

    def update_ui(self, move_point=True):
        if self.ui_mode == 'none':
            return False
        is_predict = False
        snap_qcolor = self.calibrate_color(self.user_color, self.pos)
        self.color = snap_qcolor
        # self.emit(SIGNAL('update_color'), str('background-color: %s' % self.color.name()))
        self.update_color.emit(str('background-color: %s' % self.color.name()))

        if self.ui_mode == 'point':
            if move_point:
                self.uiControl.movePoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
            else:
                self.user_color, self.brushWidth, isNew = self.uiControl.addPoint(self.pos, snap_qcolor,
                                                                                  self.user_color, self.brushWidth)
                if isNew:
                    is_predict = True
                    # self.predict_color()

        if self.ui_mode == 'stroke':
            self.uiControl.addStroke(self.prev_pos, self.pos, snap_qcolor, self.user_color, self.brushWidth)
        if self.ui_mode == 'erase':
            isRemoved = self.uiControl.erasePoint(self.pos)
            if isRemoved:
                is_predict = True
                # self.predict_color()
        return is_predict

    def reset(self):
        self.ui_mode = 'none'
        self.pos = None
        self.result = None
        self.user_color = None
        self.color = None
        self.uiControl.reset()
        self.init_color()
        self.update_result.emit(None)
        self.update()

    def scale_point(self, pnt):
        x = int((pnt.x() - self.dw) / float(self.win_w) * self.load_size)
        y = int((pnt.y() - self.dh) / float(self.win_h) * self.load_size)
        return x, y

    def valid_point(self, pnt):
        if pnt is None:
            print('WARNING: no point\n')
            return None
        else:
            if pnt.x() >= self.dw and pnt.y() >= self.dh and pnt.x() < self.win_size - self.dw and pnt.y() < self.win_size - self.dh:
                x = int(np.round(pnt.x()))
                y = int(np.round(pnt.y()))
                return QPoint(x, y)
            else:
                print('WARNING: invalid point (%d, %d)\n' % (pnt.x(), pnt.y()))
                return None

    def init_color(self):
        self.user_color = QColor(128, 128, 128)  # default color red
        self.color = self.user_color

    def change_color(self, pos=None):
        if pos is not None:
            x, y = self.scale_point(pos)
            L = self.im_lab[y, x, 0]
            # self.emit(SIGNAL('update_gamut'), L)
            self.update_gammut.emit(L)

            used_colors = self.uiControl.used_colors()
            # self.emit(SIGNAL('used_colors'), used_colors)
            self.used_colors.emit(used_colors)

            snap_color = self.calibrate_color(self.user_color, pos)
            c = np.array((snap_color.red(), snap_color.green(), snap_color.blue()), np.uint8)
            # self.emit(SIGNAL('update_ab'), c)
            self.update_ab.emit(c)

    def calibrate_color(self, c, pos):
        x, y = self.scale_point(pos)

        # snap color based on L color
        color_array = np.array((c.red(), c.green(), c.blue())).astype(
            'uint8')
        mean_L = self.im_l[y, x]
        snap_color = snap_ab(mean_L, color_array)
        snap_qcolor = QColor(snap_color[0], snap_color[1], snap_color[2])
        return snap_qcolor

    def set_color(self, c_rgb):
        c = QColor(c_rgb[0], c_rgb[1], c_rgb[2])
        self.user_color = c
        snap_qcolor = self.calibrate_color(c, self.pos)
        self.color = snap_qcolor
        # self.emit(SIGNAL('update_color'), str('background-color: %s' % self.color.name()))
        self.update_color.emit(str('background-color: %s' % self.color.name()))
        self.uiControl.update_color(snap_qcolor, self.user_color)
        self.compute_changed_color()

    def erase(self):
        self.eraseMode = not self.eraseMode

    def load_image(self):
        img_path = QFileDialog.getOpenFileName(self, 'load an input image')[0]
        if img_path is not None and os.path.exists(img_path):
            self.init_result(img_path)

    def apply_image(self):
        self.compute_result()

    def save_result(self):
        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)
        print(path)
        # add
        #########
        # original size image
        #########
        org_ab = cv2.resize(self.ab, (self.im_full.shape[1], self.im_full.shape[0]),
                            interpolation=cv2.INTER_AREA)  # INTER_CUBIC
        org_ab = org_ab * 110
        org_pred_lab = np.concatenate((self.org_im_l, org_ab), axis=2)
        org_pred_lab = (np.clip(color.lab2rgb(org_pred_lab), 0, 1) * 255.)

        #         if mode == "ours":
        #             saved_rgb = self.org_my_results * 0.5 + org_pred_lab * 0.5
        #         elif mode == "nohint":
        #             saved_rgb = self.org_my_results

        saved_rgb = self.org_my_results * 0.5 + org_pred_lab * 0.5
        # saved_rgb = self.org_my_results

        self.result = saved_rgb.astype('uint8')
        #
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        # save_path = "_".join([path, suffix])
        # save_path = os.path.join('/'.join(path.split('/')[:-1]), 'output_img') ##

        fileSave = QFileDialog.getSaveFileName(self, "save an output image", "./")
        fileName = fileSave[0].split('/')[-1]
        save_path = fileSave[0][0:-len(fileName) - 1]

        print('saving result to <%s>\n' % save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255
        # cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
        # cv2.imwrite(os.path.join(save_path, 'ours.png'), result_bgr)

        cv2.imwrite(os.path.join(save_path, f'{fileName}.png'), result_bgr)

    def enable_gray(self):
        self.use_gray = not self.use_gray
        self.update()

    def compute_changed_color(self):
        im, mask = self.uiControl.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1))  # (1, H, W)
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))  # (3, H, W)
        self.im_ab0 = im_lab[1:3, :, :]

        # _im_lab is 1) normalized 2) a torch tensor
        _im_lab = self.im_lab.transpose((2, 0, 1))
        _im_lab = np.concatenate(((_im_lab[[0], :, :] - 50) / 100, _im_lab[1:, :, :] / 110), axis=0)
        _im_lab = torch.from_numpy(_im_lab).type(torch.FloatTensor).to(self.device)

        # _img_mask is 1) normalized ab 2) flipped mask
        _img_mask = np.concatenate((self.im_ab0 / 110, (255 - self.im_mask0) / 255), axis=0)
        _img_mask = torch.from_numpy(_img_mask).type(torch.FloatTensor).to(self.device)

        # _im_lab is the full color image, _img_mask is the ab_hint+mask
        ab = self.model(_im_lab.unsqueeze(0), _img_mask.unsqueeze(0))
        ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                       h=self.load_size // self.model.patch_size, w=self.load_size // self.model.patch_size,
                       p1=self.model.patch_size, p2=self.model.patch_size)[0]
        ab = ab.detach().numpy()
        self.ab = ab

        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_AREA)  # INTER_CUBIC
        ab_win = ab_win * 110
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        #########
        # my model
        #########
        my_results = cv2.resize(self.my_results, (self.win_w, self.win_h), interpolation=cv2.INTER_AREA).astype(
            np.float32)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255.)
        # pred_rgb = my_results

        print(f"current mode {mode}")
        if mode == "ours":
            pred_rgb = my_results * 0.5 + pred_rgb * 0.5
        elif mode == "nohint":
            pred_rgb = my_results

        pred_rgb = pred_rgb.astype('uint8')
        #####################################################
        self.result = pred_rgb
        # self.emit(SIGNAL('update_result'), self.result)
        # self.update_result.emit(self.result)
        self.update()

    def compute_result(self):
        im, mask = self.uiControl.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1))  # (1, H, W)
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))  # (3, H, W)
        self.im_ab0 = im_lab[1:3, :, :]

        # _im_lab is 1) normalized 2) a torch tensor
        _im_lab = self.im_lab.transpose((2, 0, 1))
        _im_lab = np.concatenate(((_im_lab[[0], :, :] - 50) / 100, _im_lab[1:, :, :] / 110), axis=0)
        _im_lab = torch.from_numpy(_im_lab).type(torch.FloatTensor).to(self.device)

        # _img_mask is 1) normalized ab 2) flipped mask
        _img_mask = np.concatenate((self.im_ab0 / 110, (255 - self.im_mask0) / 255), axis=0)
        _img_mask = torch.from_numpy(_img_mask).type(torch.FloatTensor).to(self.device)

        # _im_lab is the full color image, _img_mask is the ab_hint+mask
        ab = self.model(_im_lab.unsqueeze(0), _img_mask.unsqueeze(0))
        ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                       h=self.load_size // self.model.patch_size, w=self.load_size // self.model.patch_size,
                       p1=self.model.patch_size, p2=self.model.patch_size)[0]
        ab = ab.detach().numpy()
        self.ab = ab

        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_AREA)  # INTER_CUBIC
        ab_win = ab_win * 110
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        #########
        # my model
        #########
        my_results = cv2.resize(self.my_results, (self.win_w, self.win_h), interpolation=cv2.INTER_AREA).astype(
            np.float32)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255.)
        # pred_rgb = my_results

        print(f"current mode {mode}")
        if mode == "ours":
            pred_rgb = my_results * 0.5 + pred_rgb * 0.5
        elif mode == "nohint":
            pred_rgb = my_results

        pred_rgb = pred_rgb.astype('uint8')
        #####################################################
        self.result = pred_rgb
        # self.emit(SIGNAL('update_result'), self.result)
        self.update_result.emit(self.result)
        self.update()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), QColor(49, 54, 49))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.use_gray or self.result is None:
            im = self.gray_win
        else:
            im = self.result

        if im is not None:
            qImg = QImage(im.tobytes(), im.shape[1], im.shape[0], QImage.Format.Format_RGB888)
            painter.drawImage(self.dw, self.dh, qImg)

        self.uiControl.update_painter(painter)
        painter.end()

    # def wheelEvent(self, event):
    #     d = event.delta() / 120
    #     self.brushWidth = min(4.05 * self.scale, max(0, self.brushWidth + d * self.scale))
    #     print('update brushWidth = %f' % self.brushWidth)
    #     self.update_ui(move_point=True)
    #     self.update()

    def is_same_point(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return False
        dx = pos1.x() - pos2.x()
        dy = pos1.y() - pos2.y()
        d = dx * dx + dy * dy
        # print('distance between points = %f' % d)
        return d < 25

    def mousePressEvent(self, event):
        print('mouse press', event.pos())
        pos = self.valid_point(event.pos())

        if pos is not None:
            if event.button() == Qt.LeftButton:
                self.pos = pos
                self.ui_mode = 'point'
                self.change_color(pos)
                self.update_ui(move_point=False)
                self.compute_changed_color()

            if event.button() == Qt.RightButton:
                # draw the stroke
                self.pos = pos
                self.ui_mode = 'erase'
                self.update_ui(move_point=False)
                self.compute_changed_color()

    def mouseMoveEvent(self, event):
        self.pos = self.valid_point(event.pos())
        if self.pos is not None:
            if self.ui_mode == 'point':
                self.update_ui(move_point=True)
                self.compute_changed_color()

    def mouseReleaseEvent(self, event):
        pass

    def sizeHint(self):
        return QSize(self.win_size, self.win_size)  # 28 * 8


class GUIPalette(QWidget):
    update_color = pyqtSignal(object)

    def __init__(self, grid_sz=(6, 3)):
        QWidget.__init__(self)
        self.color_width = 25
        self.border = 6
        self.win_width = grid_sz[0] * self.color_width + (grid_sz[0] + 1) * self.border
        self.win_height = grid_sz[1] * self.color_width + (grid_sz[1] + 1) * self.border
        self.setFixedSize(self.win_width, self.win_height)
        self.num_colors = grid_sz[0] * grid_sz[1]
        self.grid_sz = grid_sz
        self.colors = None
        self.color_id = -1
        self.reset()

    def set_colors(self, colors):
        if colors is not None:
            self.colors = (colors[:min(colors.shape[0], self.num_colors), :] * 255).astype(np.uint8)
            self.color_id = -1
            self.update()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(event.rect(), Qt.GlobalColor.white)
        if self.colors is not None:
            for n, c in enumerate(self.colors):
                ca = QColor(c[0], c[1], c[2], 255)
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.setBrush(ca)
                grid_x = n % self.grid_sz[0]
                grid_y = (n - grid_x) // self.grid_sz[0]
                x = grid_x * (self.color_width + self.border) + self.border
                y = grid_y * (self.color_width + self.border) + self.border

                if n == self.color_id:
                    painter.drawEllipse(x, y, self.color_width, self.color_width)
                else:
                    painter.drawRoundedRect(x, y, self.color_width, self.color_width, 2, 2)

        painter.end()

    def sizeHint(self):
        return QSize(self.win_width, self.win_height)

    def reset(self):
        self.colors = None
        self.mouseClicked = False
        self.color_id = -1
        self.update()

    def selected_color(self, pos):
        width = self.color_width + self.border
        dx = pos.x() % width
        dy = pos.y() % width
        if dx >= self.border and dy >= self.border:
            x_id = (pos.x() - dx) // width
            y_id = (pos.y() - dy) // width
            color_id = x_id + y_id * self.grid_sz[0]
            return int(color_id)
        else:
            return -1

    def update_ui(self, color_id):
        self.color_id = int(color_id)
        self.update()
        if color_id >= 0 and self.colors is not None:
            print('choose color (%d) type (%s)' % (color_id, type(color_id)))
            color = self.colors[color_id]
            # self.emit(SIGNAL('update_color'), color)
            self.update_color.emit(color)
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:  # click the point
            color_id = self.selected_color(event.pos())
            self.update_ui(color_id)
            self.mouseClicked = True

    def mouseMoveEvent(self, event):
        if self.mouseClicked:
            color_id = self.selected_color(event.pos())
            self.update_ui(color_id)

    def mouseReleaseEvent(self, event):
        self.mouseClicked = False


class GUI_VIS(QWidget):
    def __init__(self, win_size=256, scale=2.0):
        QWidget.__init__(self)
        self.result = None
        self.win_width = win_size
        self.win_height = win_size
        self.scale = scale
        self.setFixedSize(self.win_width, self.win_height)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(event.rect(), QColor(49, 54, 49))
        if self.result is not None:
            h, w, c = self.result.shape
            qImg = QImage(self.result.tostring(), w, h, QImage.Format.Format_RGB888)
            dw = int((self.win_width - w) // 2)
            dh = int((self.win_height - h) // 2)
            painter.drawImage(dw, dh, qImg)

        painter.end()

    def update_result(self, result):
        self.result = result
        self.update()

    def sizeHint(self):
        return QSize(self.win_width, self.win_height)

    def reset(self):
        self.update()
        self.result = None

    def is_valid_point(self, pos):
        if pos is None:
            return False
        else:
            x = pos.x()
            y = pos.y()
            return x >= 0 and y >= 0 and x < self.win_width and y < self.win_height

    def scale_point(self, pnt):
        x = int(pnt.x() / self.scale)
        y = int(pnt.y() / self.scale)
        return x, y

    def mousePressEvent(self, event):
        pos = event.pos()
        x, y = self.scale_point(pos)
        if event.button() == Qt.LeftButton and self.is_valid_point(pos):  # click the point
            if self.result is not None:
                color = self.result[y, x, :]  #
                print('color', color)

    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass


class UserEdit(object):
    def __init__(self, mode, win_size, load_size, img_size):
        self.mode = mode
        self.win_size = win_size
        self.img_size = img_size
        self.load_size = load_size
        print('image_size', self.img_size)
        max_width = np.max(self.img_size)
        self.scale = float(max_width) / self.load_size  # original image to 224 ration
        self.dw = int((self.win_size - img_size[0]) // 2)
        self.dh = int((self.win_size - img_size[1]) // 2)
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.ui_count = 0
        print(self)

    def scale_point(self, in_x, in_y, w):
        x = int((in_x - self.dw) / float(self.img_w) * self.load_size) + w
        y = int((in_y - self.dh) / float(self.img_h) * self.load_size) + w
        return x, y

    def __str__(self):
        return "add (%s) with win_size %3.3f, load_size %3.3f" % (self.mode, self.win_size, self.load_size)


class PointEdit(UserEdit):
    def __init__(self, win_size, load_size, img_size):
        UserEdit.__init__(self, 'point', win_size, load_size, img_size)

    def add(self, pnt, color, userColor, width, ui_count):
        self.pnt = pnt
        self.color = color
        self.userColor = userColor
        self.width = width
        self.ui_count = ui_count

    def select_old(self, pnt, ui_count):
        self.pnt = pnt
        self.ui_count = ui_count
        return self.userColor, self.width

    def update_color(self, color, userColor):
        self.color = color
        self.userColor = userColor

    def updateInput(self, im, mask, vis_im):
        w = int(self.width / self.scale)
        pnt = self.pnt
        x1, y1 = self.scale_point(pnt.x(), pnt.y(), -w)
        tl = (x1, y1)
        # x2, y2 = self.scale_point(pnt.x(), pnt.y(), w)
        # br = (x2, y2)
        br = (x1 + 1, y1 + 1)  # hint size fixed to 2
        c = (self.color.red(), self.color.green(), self.color.blue())
        uc = (self.userColor.red(), self.userColor.green(), self.userColor.blue())
        cv2.rectangle(mask, tl, br, 255, -1)
        cv2.rectangle(im, tl, br, c, -1)
        cv2.rectangle(vis_im, tl, br, uc, -1)

    def is_same(self, pnt):
        dx = abs(self.pnt.x() - pnt.x())
        dy = abs(self.pnt.y() - pnt.y())
        return dx <= self.width + 1 and dy <= self.width + 1

    def update_painter(self, painter):
        w = max(3, self.width)
        c = self.color
        r = c.red()
        g = c.green()
        b = c.blue()
        ca = QColor(c.red(), c.green(), c.blue(), 255)
        d_to_black = r * r + g * g + b * b
        d_to_white = (255 - r) * (255 - r) + (255 - g) * (255 - g) + (255 - r) * (255 - r)
        if d_to_black > d_to_white:
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
        else:
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.setBrush(ca)
        painter.drawRoundedRect(self.pnt.x() - w, self.pnt.y() - w, 1 + 2 * w, 1 + 2 * w, 2, 2)


class UIControl:
    def __init__(self, win_size=256, load_size=224):
        self.win_size = win_size
        self.load_size = load_size
        self.reset()
        self.userEdit = None
        self.userEdits = []
        self.ui_count = 0

    def setImageSize(self, img_size):
        self.img_size = img_size

    def addStroke(self, prevPnt, nextPnt, color, userColor, width):
        pass

    def erasePoint(self, pnt):
        isErase = False
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdits.remove(ue)
                print('remove user edit %d\n' % id)
                isErase = True
                break
        return isErase

    def addPoint(self, pnt, color, userColor, width):
        self.ui_count += 1
        print('process add Point')
        self.userEdit = None
        isNew = True
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdit = ue
                isNew = False
                print('select user edit %d\n' % id)
                break

        if self.userEdit is None:
            self.userEdit = PointEdit(self.win_size, self.load_size, self.img_size)
            self.userEdits.append(self.userEdit)
            print('add user edit %d\n' % len(self.userEdits))
            self.userEdit.add(pnt, color, userColor, width, self.ui_count)
            return userColor, width, isNew
        else:
            userColor, width = self.userEdit.select_old(pnt, self.ui_count)
            return userColor, width, isNew

    def movePoint(self, pnt, color, userColor, width):
        self.userEdit.add(pnt, color, userColor, width, self.ui_count)

    def update_color(self, color, userColor):
        self.userEdit.update_color(color, userColor)

    def update_painter(self, painter):
        for ue in self.userEdits:
            if ue is not None:
                ue.update_painter(painter)

    def get_stroke_image(self, im):
        return im

    def used_colors(self):  # get recently used colors
        if len(self.userEdits) == 0:
            return None
        nEdits = len(self.userEdits)
        ui_counts = np.zeros(nEdits)
        ui_colors = np.zeros((nEdits, 3))
        for n, ue in enumerate(self.userEdits):
            ui_counts[n] = ue.ui_count
            c = ue.userColor
            ui_colors[n, :] = [c.red(), c.green(), c.blue()]

        ui_counts = np.array(ui_counts)
        ids = np.argsort(-ui_counts)
        ui_colors = ui_colors[ids, :]
        unique_colors = []
        for ui_color in ui_colors:
            is_exit = False
            for u_color in unique_colors:
                d = np.sum(np.abs(u_color - ui_color))
                if d < 0.1:
                    is_exit = True
                    break

            if not is_exit:
                unique_colors.append(ui_color)

        unique_colors = np.vstack(unique_colors)
        return unique_colors / 255.0

    def get_input(self):
        h = self.load_size
        w = self.load_size
        im = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)
        vis_im = np.zeros((h, w, 3), np.uint8)

        for ue in self.userEdits:
            ue.updateInput(im, mask, vis_im)

        return im, mask

    def reset(self):
        self.userEdits = []
        self.userEdit = None
        self.ui_count = 0


def qcolor2lab_1d(qc):
    # take 1d numpy array and do color conversion
    c = np.array([qc.red(), qc.green(), qc.blue()], np.uint8)
    return rgb2lab_1d(c)


def rgb2lab_1d(in_rgb):
    # take 1d numpy array and do color conversion
    # print('in_rgb', in_rgb)
    return color.rgb2lab(in_rgb[np.newaxis, np.newaxis, :]).flatten()


def lab2rgb_1d(in_lab, clip=True, dtype='uint8'):
    warnings.filterwarnings("ignore")
    tmp_rgb = color.lab2rgb(in_lab[np.newaxis, np.newaxis, :]).flatten()
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    if dtype == 'uint8':
        tmp_rgb = np.round(tmp_rgb * 255).astype('uint8')
    return tmp_rgb


def snap_ab(input_l, input_rgb, return_type='rgb'):
    ''' given an input lightness and rgb, snap the color into a region where l,a,b is in-gamut
    '''
    T = 20
    warnings.filterwarnings("ignore")
    input_lab = rgb2lab_1d(np.array(input_rgb))  # convert input to lab
    conv_lab = input_lab.copy()  # keep ab from input
    for t in range(T):
        conv_lab[0] = input_l  # overwrite input l with input ab
        old_lab = conv_lab
        tmp_rgb = color.lab2rgb(conv_lab[np.newaxis, np.newaxis, :]).flatten()
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
        conv_lab = color.rgb2lab(tmp_rgb[np.newaxis, np.newaxis, :]).flatten()
        dif_lab = np.sum(np.abs(conv_lab - old_lab))
        if dif_lab < 1:
            break
        # print(conv_lab)

    conv_rgb_ingamut = lab2rgb_1d(conv_lab, clip=True, dtype='uint8')
    if (return_type == 'rgb'):
        return conv_rgb_ingamut

    elif (return_type == 'lab'):
        conv_lab_ingamut = rgb2lab_1d(conv_rgb_ingamut)
        return conv_lab_ingamut


class abGrid():
    def __init__(self, gamut_size=110, D=1):
        self.D = D
        self.vals_b, self.vals_a = np.meshgrid(np.arange(-gamut_size, gamut_size + D, D),
                                               np.arange(-gamut_size, gamut_size + D, D))
        self.pts_full_grid = np.concatenate((self.vals_a[:, :, np.newaxis], self.vals_b[:, :, np.newaxis]), axis=2)
        self.A = self.pts_full_grid.shape[0]
        self.B = self.pts_full_grid.shape[1]
        self.AB = self.A * self.B
        self.gamut_size = gamut_size

    def update_gamut(self, l_in):
        warnings.filterwarnings("ignore")
        thresh = 1.0
        pts_lab = np.concatenate((l_in + np.zeros((self.A, self.B, 1)), self.pts_full_grid), axis=2)
        self.pts_rgb = (255 * np.clip(color.lab2rgb(pts_lab), 0, 1)).astype('uint8')
        pts_lab_back = color.rgb2lab(self.pts_rgb)
        pts_lab_diff = np.linalg.norm(pts_lab - pts_lab_back, axis=2)

        self.mask = pts_lab_diff < thresh
        mask3 = np.tile(self.mask[..., np.newaxis], [1, 1, 3])
        self.masked_rgb = self.pts_rgb.copy()
        self.masked_rgb[np.invert(mask3)] = 255
        return self.masked_rgb, self.mask

    def ab2xy(self, a, b):
        y = self.gamut_size + a
        x = self.gamut_size + b
        # print('ab2xy (%d, %d) -> (%d, %d)' % (a, b, x, y))
        return x, y

    def xy2ab(self, x, y):
        a = y - self.gamut_size
        b = x - self.gamut_size
        # print('xy2ab (%d, %d) -> (%d, %d)' % (x, y, a, b))
        return a, b


class IColoriTUI(QWidget):
    def __init__(self, color_model, nohint_model=None, img_file=None, load_size=224, win_size=256, device='cpu'):
        # draw the layout
        QWidget.__init__(self)

        # main layout
        mainLayout = QHBoxLayout()
        self.setLayout(mainLayout)

        # gamut layout
        self.gamutWidget = GUIGamut(gamut_size=110)
        gamutLayout = self.AddWidget(self.gamutWidget, 'ab Color Gamut')
        colorLayout = QVBoxLayout()

        colorLayout.addLayout(gamutLayout)
        mainLayout.addLayout(colorLayout)

        # palette
        self.usedPalette = GUIPalette(grid_sz=(10, 1))
        upLayout = self.AddWidget(self.usedPalette, 'Recently used colors')
        colorLayout.addLayout(upLayout)

        self.colorPush = QPushButton()  # to visualize the selected color
        self.colorPush.setFixedWidth(self.usedPalette.width())
        self.colorPush.setFixedHeight(25)
        self.colorPush.setStyleSheet("background-color: grey")
        colorPushLayout = self.AddWidget(self.colorPush, 'Current Color')
        colorLayout.addLayout(colorPushLayout)
        colorLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        ###################################
        ## colorize 버튼 추가
        ###################################

        self.colorize_ours = QPushButton("&hint + no hint (default)")  # colorize as our method
        self.colorize_ours.setFixedWidth(self.usedPalette.width() * 0.75)
        self.colorize_ours.setFixedHeight(35)

        self.colorize_hint = QPushButton("&hint (iColoriT)")  # colorize hint
        self.colorize_hint.setFixedWidth(self.usedPalette.width() * 0.75)
        self.colorize_hint.setFixedHeight(35)

        self.colorize_nohint = QPushButton("&no hint (Our Model)")  # colorize nohint
        self.colorize_nohint.setFixedWidth(self.usedPalette.width() * 0.75)
        self.colorize_nohint.setFixedHeight(35)

        self.Colorize_Menu = QVBoxLayout()
        self.Colorize_Menu.setSpacing(30)
        self.Colorize_Menu.addWidget(self.colorize_ours)
        self.Colorize_Menu.addWidget(self.colorize_hint)
        self.Colorize_Menu.addWidget(self.colorize_nohint)
        self.Colorize_Menu.setAlignment(Qt.AlignmentFlag.AlignCenter)

        groupBox = QGroupBox("Colorization Mode")
        groupBox.setLayout(self.Colorize_Menu)
        colorLayout.addWidget(groupBox)

        self.colorize_ours.clicked.connect(lambda: self.reset_mode("ours"))
        self.colorize_hint.clicked.connect(lambda: self.reset_mode("hint"))
        self.colorize_nohint.clicked.connect(lambda: self.reset_mode("nohint"))

        ###################################
        self.bApply = QPushButton('&Colorize')
        self.bApply.setStyleSheet('QPushButton {background-color: #A3C1DA; color: white; font-size: 20px; '
                                  'font-family: Times;}')
        self.bApply.setToolTip('apply the final image')

        colorLayout.addWidget(self.bApply)
        ###################################

        # colorLayout.setAlignment(Qt.AlignCenter)

        # drawPad layout
        drawPadLayout = QVBoxLayout()
        mainLayout.addLayout(drawPadLayout)
        # self.drawWidget = GUIDraw(color_model, load_size=load_size, win_size=win_size, device=device)
        self.drawWidget = GUIDraw(color_model, nohint_model=nohint_model, load_size=224, win_size=512, device=device)

        drawPadLayout = self.AddWidget(self.drawWidget, 'Drawing Pad')
        mainLayout.addLayout(drawPadLayout)

        drawPadMenu = QHBoxLayout()

        self.bGray = QCheckBox("&Gray")
        self.bGray.setToolTip('show gray-scale image')

        self.bLoad = QPushButton('&Load')
        self.bLoad.setStyleSheet('QPushButton {background-color: #A3C1DA; color: black;}')
        self.bLoad.setToolTip('load an input image')
        self.bSave = QPushButton("&Save Result")
        self.bSave.setStyleSheet('QPushButton {background-color: #A3C1DA; color: black;}')
        self.bSave.setToolTip('Save the current result.')

        drawPadMenu.addWidget(self.bGray)
        drawPadMenu.addWidget(self.bLoad)
        drawPadMenu.addWidget(self.bSave)

        drawPadLayout.addLayout(drawPadMenu)
        self.visWidget = GUI_VIS(win_size=512, scale=win_size / float(load_size))
        visWidgetLayout = self.AddWidget(self.visWidget, 'Colorized Result')
        mainLayout.addLayout(visWidgetLayout)

        self.bRestart = QPushButton("&Restart")
        self.bRestart.setStyleSheet('QPushButton {background-color: #A3C1DA; color: black;}')
        self.bRestart.setToolTip('Restart the system')

        self.bQuit = QPushButton("&Quit")
        self.bQuit.setStyleSheet('QPushButton {background-color: #A3C1DA; color: black;}')
        self.bQuit.setToolTip('Quit the system.')
        visWidgetMenu = QHBoxLayout()
        visWidgetMenu.addWidget(self.bRestart)

        visWidgetMenu.addWidget(self.bQuit)
        visWidgetLayout.addLayout(visWidgetMenu)

        self.drawWidget.update()
        self.visWidget.update()
        # self.colorPush.clicked.connect(self.drawWidget.change_color)

        # color indicator
        self.drawWidget.update_color.connect(self.colorPush.setStyleSheet)

        # update result
        self.drawWidget.update_result.connect(self.visWidget.update_result)  # pyqt5

        # update gamut
        self.drawWidget.update_gammut.connect(self.gamutWidget.set_gamut)  # pyqt5
        self.drawWidget.update_ab.connect(self.gamutWidget.set_ab)
        self.gamutWidget.update_color.connect(self.drawWidget.set_color)

        # connect palette
        self.drawWidget.used_colors.connect(self.usedPalette.set_colors)  # pyqt5
        self.usedPalette.update_color.connect(self.drawWidget.set_color)
        self.usedPalette.update_color.connect(self.gamutWidget.set_ab)

        # menu events
        self.bGray.setChecked(True)
        self.bRestart.clicked.connect(self.reset)
        self.bQuit.clicked.connect(self.quit)
        self.bGray.toggled.connect(self.enable_gray)
        self.bSave.clicked.connect(self.save)
        self.bLoad.clicked.connect(self.load)
        self.bApply.clicked.connect(self.apply)

        self.start_t = time.time()

        if img_file is not None:
            self.drawWidget.init_result(img_file)
        print('UI initialized')

    def AddWidget(self, widget, title):
        widgetLayout = QVBoxLayout()
        widgetBox = QGroupBox()
        widgetBox.setTitle(title)
        vbox_t = QVBoxLayout()
        vbox_t.addWidget(widget, alignment=Qt.AlignmentFlag.AlignCenter)
        widgetBox.setLayout(vbox_t)
        widgetLayout.addWidget(widgetBox)

        return widgetLayout

    def nextImage(self):
        self.drawWidget.nextImage()

    def reset(self):
        # self.start_t = time.time()
        print('============================reset all=========================================')
        self.visWidget.reset()
        self.gamutWidget.reset()
        self.usedPalette.reset()
        self.drawWidget.reset()
        self.update()
        self.colorPush.setStyleSheet("background-color: grey")

    def enable_gray(self):
        self.drawWidget.enable_gray()

    def quit(self):
        print('time spent = %3.3f' % (time.time() - self.start_t))
        self.close()

    def apply(self):
        self.drawWidget.apply_image()

    def save(self):
        print('time spent = %3.3f' % (time.time() - self.start_t))
        self.drawWidget.save_result()

    def load(self):
        self.drawWidget.load_image()

    def change_color(self):
        print('change color')
        self.drawWidget.change_color(use_suggest=True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.reset()

        if event.key() == Qt.Key_Q:
            self.save()
            self.quit()

        if event.key() == Qt.Key_S:
            self.save()

        if event.key() == Qt.Key_G:
            self.bGray.toggle()

        if event.key() == Qt.Key_L:
            self.load()

    ######################################
    ##### reset colorization mode #######
    ######################################
    def reset_mode(self, mode):
        # self.start_t = time.time()
        mode_info(str(mode))
        print('============================reset mode=========================================')
        self.reset()
        # print(mode)

    ######################################
    ######################################


def get_args():
    parser = argparse.ArgumentParser('Colorization UI', add_help=False)
    # Directories
    parser.add_argument('--model_path', type=str, default='path/to/checkpoints', help='checkpoint path of model')
    parser.add_argument('--target_image', default='./test_img/41.jpg', type=str, help='validation dataset path')
    parser.add_argument('--device', default='cpu', help='device to use for testing')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')

    # Model parameters
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str, help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate (default: 0.1)')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode')
    parser.add_argument('--mask_cent', action='store_true', help='mask_cent')

    args = parser.parse_args()

    return args


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_rpb=args.use_rpb,
        avg_hint=args.avg_hint,
        head_mode=args.head_mode,
        mask_cent=args.mask_cent,
    )

    return model



def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # relative positional bias option
        self.use_rpb = use_rpb
        if use_rpb:
            self.window_size = window_size
            self.rpb_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            trunc_normal_(self.rpb_table, std=.02)

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
            coords_flatten = torch.flatten(coords, 1)  # 2, h*w
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # h*w, h*w
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.use_rpb:
            relative_position_bias = self.rpb_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # h*w,h*w,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w, h*w
            attn += relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_rpb=use_rpb, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, mask_cent=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_cent = mask_cent

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if self.mask_cent:
            x[:, -1] = x[:, -1] - 0.5
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

##################################### Colorization #################################


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CnnHead(nn.Module):
    def __init__(self, embed_dim, num_classes, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.window_size = window_size

        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = rearrange(x, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
        x = self.head(x)
        x = rearrange(x, 'b c p1 p2 -> b (p1 p2) c')
        return x


class LocalAttentionHead(nn.Module):
    def __init__(
            self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # masking attn
        mask = torch.ones((window_size**2, window_size**2))
        kernel_size = 3
        for i in range(window_size):
            for j in range(window_size):
                cur_map = torch.ones((window_size, window_size))
                stx, sty = max(i - kernel_size // 2, 0), max(j - kernel_size // 2, 0)
                edx, edy = min(i + kernel_size // 2, window_size - 1), min(j + kernel_size // 2, window_size - 1)
                cur_map[stx:edx + 1, sty:edy + 1] = 0
                cur_map = cur_map.flatten()
                mask[i * window_size + j] = cur_map
        self.register_buffer('mask', mask)

        # relative positional bias option
        self.use_rpb = use_rpb
        if use_rpb:
            self.window_size = window_size
            self.rpb_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            trunc_normal_(self.rpb_table, std=.02)

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
            coords_flatten = torch.flatten(coords, 1)  # 2, h*w
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # h*w, h*w
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # masking attn
        mask_value = max_neg_value(attn)
        attn.masked_fill_(self.mask.bool(), mask_value)

        if self.use_rpb:
            relative_position_bias = self.rpb_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # h*w,h*w,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w, h*w
            attn += relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class IColoriT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=512, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_rpb=False, avg_hint=False, head_mode='default', mask_cent=False):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 2 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.avg_hint = avg_hint

        # self.mask_token = nn.Parameter(torch.zeros(2))
        # trunc_normal_(self.mask_token, std=.02)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, mask_cent=mask_cent)
        num_patches = self.patch_embed.num_patches  # 2

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            init_values=init_values, use_rpb=use_rpb, window_size=img_size // patch_size)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        if head_mode == 'linear':
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif head_mode == 'cnn':
            self.head = CnnHead(embed_dim, num_classes, window_size=img_size // patch_size)
        elif head_mode == 'locattn':
            self.head = LocalAttentionHead(embed_dim, num_classes, window_size=img_size // patch_size)
        else:
            raise NotImplementedError('Check head type')

        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        # mask is 1D of 2D if 2D
        B, _, H, W = x.shape
        if mask.dim() == 2:
            _, L = mask.shape
            # assume square inputs
            hint_size = int(math.sqrt(H * W // L))
            _device = '.cuda' if x.device.type == 'cuda' else ''

            # hint location = 0, non-hint location = 1
            mask = torch.reshape(mask, (B, H // hint_size, W // hint_size))
            _mask = mask.unsqueeze(1).type(f'torch{_device}.FloatTensor')
            _full_mask = F.interpolate(_mask, scale_factor=hint_size)  # Needs to be Float
            full_mask = _full_mask.type(f'torch{_device}.BoolTensor')

            # mask ab channels
            if self.avg_hint:
                _avg_x = F.interpolate(x, size=(H // hint_size, W // hint_size), mode='bilinear')
                _avg_x[:, 1, :, :].masked_fill_(mask.squeeze(1), 0)
                _avg_x[:, 2, :, :].masked_fill_(mask.squeeze(1), 0)
                x_ab = F.interpolate(_avg_x, scale_factor=hint_size, mode='nearest')[:, 1:, :, :]
                x = torch.cat((x[:, 0, :, :].unsqueeze(1), x_ab), dim=1)
            else:
                x[:, 1, :, :].masked_fill_(full_mask.squeeze(1), 0)
                x[:, 2, :, :].masked_fill_(full_mask.squeeze(1), 0)
        elif mask.dim() == 4:  # (B, 3, H, W) Zhang mode. channel 0,1 is ab. channel2 is mask
            # TODO ONLY FOR DEMO
            _device = '.cuda' if x.device.type == 'cuda' else ''
            _full_mask_ab = mask.type(f'torch{_device}.FloatTensor')[:, :2, :, :]
            x[:, 1:, :, :] = _full_mask_ab
            _full_mask = mask[:, 2, :, :].unsqueeze(1).type(f'torch{_device}.FloatTensor')
        else:
            raise NotImplementedError('Check the mask dimension')

        if self.in_chans == 4:
            x = torch.cat((x, 1 - _full_mask), dim=1)

        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()  # (B, 14*14, 768)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        x = self.tanh(x)
        return x


@register_model
def icolorit_tiny_4ch_patch8_224(pretrained=False, **kwargs):
    model = IColoriT(
        num_classes=128,
        img_size=224,
        patch_size=8,
        in_chans=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def icolorit_tiny_4ch_patch16_224(pretrained=False, **kwargs):
    model = IColoriT(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def icolorit_tiny_4ch_patch32_224(pretrained=False, **kwargs):
    model = IColoriT(
        num_classes=2048,
        img_size=224,
        patch_size=32,
        in_chans=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def icolorit_small_4ch_patch16_224(pretrained=False, **kwargs):
    model = IColoriT(
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def icolorit_base_4ch_patch16_224(pretrained=False, **kwargs):
    model = IColoriT(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == '__main__':
    signal(SIGPIPE, SIG_DFL)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()

    model = get_model(args)
    model.to(args.device)
    checkpoint = torch.load('./saved_models/best/icolorit_base_4ch_patch16_224.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # no hint model
    print('Creating model:', 'nohint model' )
    nohint_model = smp.Unet(
            encoder_name='efficientnet-b1',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,        # model output channels (number of classes in your dataset)
            activation='sigmoid',
        )
    # 쓸모없는 키 이름 지우기
    from collections import OrderedDict
    temp_dict = OrderedDict()
    weight = torch.load('./saved_models/best/fold0_best_e49.pth', map_location='cpu')
    for k, w in weight.items():
        temp_dict[k.replace('module.model.','')] = w

    #
    nohint_model.load_state_dict(temp_dict)
    nohint_model.eval()
    #
    app = QApplication(sys.argv)
    ex = IColoriTUI(color_model=model, nohint_model=nohint_model, img_file=args.target_image,
                             load_size=args.input_size, win_size=720, device=args.device)
    ex.setWindowIcon(QIcon('gui/icon.png'))
    ex.setWindowTitle('iColoriT')
    ex.show()
    sys.exit(app.exec())
