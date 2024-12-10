import os, sys
import argparse
import pdb
from PIL import Image, ImageDraw, ImageFont
import cv2
import text_render
import image_render
import numpy as np

font_path = '' # your font path
def adjuct_font_and_width(in_text, fixed_height, ref_width, min_font_size=15, line_space=2):
    min_font_size = 15
    line_space = 2
    cur_text_width = ref_width
    cur_lab_font_size = 25
    font = ImageFont.truetype(font_path, cur_lab_font_size, encoding="utf-8")
    while True:
        success = True
        cur_x = 0
        cur_y = 0
        cur_h = 0
        for ch in in_text:
            w, h = font.getsize(ch)
            if cur_x + w >= cur_text_width:
                cur_x = 0
                cur_y += (cur_h + line_space)
                cur_h = 0
            cur_x += w
            cur_h = max(cur_h, h)
            if cur_y + cur_h >= fixed_height:
                success = False
                break
        if success is True:
            break
        else:
            if cur_lab_font_size > min_font_size:
                cur_lab_font_size -= 1
                font = ImageFont.truetype(font_path, cur_lab_font_size, encoding="utf-8")
            else:
                cur_text_width += 100
    return cur_lab_font_size, cur_text_width

def drawText(text, height, width, font, line_space=2):
    max_w = 0
    max_h = 0
    cur_x = 0
    cur_y = 0
    cur_h = 0
    img_np = np.ones((height, width, 3), dtype=np.uint8)*255
    #img = Image.new('RGB', (width, height), "#FFFFFF")
    img = Image.fromarray(img_np)
    imgDraw = ImageDraw.Draw(img)
    for ch in text:
        w, h = font.getsize(ch)
        if cur_x + w >= width:
            cur_x = 0
            cur_y += (cur_h + line_space)
            cur_h = 0
        imgDraw.text((cur_x, cur_y), ch, fill=(0,0,0), font=font)
        cur_x += w
        cur_h = max(cur_h, h)
        max_w = max([max_w, cur_x])
        max_h = max([max_h, cur_y + cur_h])
    # pdb.set_trace()
    img_np = np.array(img)
    img_out = np.ones((height, width, 3), dtype=np.uint8)*255
    #img_np[:max_h, :max_w, :]
    start_x = (width - max_w) // 2
    start_y = (height - max_h) // 2
    img_out[start_y:start_y+max_h, start_x:start_x+max_w, :] = img_np[:max_h, :max_w, :]
    return img_out
    
def adjuct_font_and_height(in_text, fixed_width, ref_height, min_font_size=15, line_space=2):
    min_font_size = 15
    line_space = 2
    cur_text_height = ref_height
    cur_lab_font_size = 25
    font = ImageFont.truetype(font_path, cur_lab_font_size, encoding="utf-8")
    while True:
        success = True
        cur_x = 0
        cur_y = 0
        cur_h = 0
        max_height = 0
        for ch in in_text:
            w, h = font.getsize(ch)
            if cur_x + w >= fixed_width:
                cur_x = 0
                cur_y += (cur_h + line_space)
                cur_h = 0
            cur_x += w
            cur_h = max(cur_h, h)
            if cur_y + cur_h >= cur_text_height:
                success = False
                break
            else:
                max_height = max([max_height, cur_y + cur_h])
        if success is True:
            break
        else:
            if cur_lab_font_size > min_font_size:
                cur_lab_font_size -= 1
                font = ImageFont.truetype(font_path, cur_lab_font_size, encoding="utf-8")
            else:
                cur_text_height += (cur_lab_font_size + line_space)
    # pdb.set_trace()
    return cur_lab_font_size, max_height



def viz_struct_res(img_path, rec_rep_dict, trans_text=None):
    # print(img_path)
    src_img = cv2.imread(img_path)
    src_h, src_w, src_c = src_img.shape

    img_pair_dict = {}
    labs_height = 0
    recs_height = 0
    tmp_width_sum = 0
    tmp_width_cnt = 0
    temp_dir = "/ps3/cv9/haowu16/Texlive/tex_temp/"
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    for key in rec_rep_dict:
        if "res_graph" in rec_rep_dict[key]:
            img_pair_dict[key] = {}
            lab_atom = rec_rep_dict[key]["lab_atom"]
            rec_atom = rec_rep_dict[key]["rec_atom"]
            
            lab_atom_img = image_render.rend(lab_atom, scale=100)
            rec_atom_img = image_render.rend(rec_atom, scale=100)
            # lab_reverse_trans = test_reverse_trans.process_trans("".join(chemfig_render.reverse_rend_text(lab_atom, scale=1, remove_dup=0, norm_circle=0, connect_distant=0)))
            # lab_atom_img = test_reverse_trans.texlive_rend(lab_reverse_trans, os.path.join(temp_dir, "{}_lab.pdf".format(image_name)))
            # rec_reverse_trans = test_reverse_trans.process_trans("".join(chemfig_render.reverse_rend_text(rec_atom, scale=1, remove_dup=0, norm_circle=0, connect_distant=0)))
            # rec_atom_img = test_reverse_trans.texlive_rend(rec_reverse_trans, os.path.join(temp_dir, "{}_rec.pdf".format(image_name)))
            # pdb.set_trace()
            img_pair_dict[key]["lab_img"] = lab_atom_img
            img_pair_dict[key]["rec_img"] = rec_atom_img
            cur_width = max([lab_atom_img.shape[1], rec_atom_img.shape[1]])
            img_pair_dict[key]["cur_width"] = cur_width
            labs_height = max([labs_height, lab_atom_img.shape[0]])
            recs_height = max([recs_height, rec_atom_img.shape[0]])
            tmp_width_sum += cur_width
            tmp_width_cnt += 1
    if tmp_width_cnt > 0:
        ref_width = int(float(tmp_width_sum)/tmp_width_cnt)
    else:
        ref_width = int(float(src_w)/len(rec_rep_dict))
    if labs_height == 0:
        labs_height = 50
    if recs_height == 0:
        recs_height = 50
    
    for key in rec_rep_dict:
        if "res_text" in rec_rep_dict[key]:
            img_pair_dict[key] = {}
            lab_text = rec_rep_dict[key]["lab_text"]
            rec_text = rec_rep_dict[key]["rec_text"]

            min_font_size = 15
            line_space = 2

            lab_font_size, lab_text_width = adjuct_font_and_width(lab_text, labs_height, ref_width)
            rec_font_size, rec_text_width = adjuct_font_and_width(rec_text, recs_height, ref_width, min_font_size=lab_font_size)
            common_font_size = min([lab_font_size, rec_font_size])
            common_text_width = max([lab_text_width, rec_text_width])

            font = ImageFont.truetype(font_path, common_font_size, encoding="utf-8")
            lab_text_img = drawText(lab_text, labs_height, common_text_width, font)
            rec_text_img = drawText(rec_text, recs_height, common_text_width, font)
            img_pair_dict[key]["lab_img"] = lab_text_img
            img_pair_dict[key]["rec_img"] = rec_text_img
            img_pair_dict[key]["cur_width"] = common_text_width

    

    img_line_space = 10
    img_col_space = 10
    result_bbox_width = 3
    out_width = 0
    for key in img_pair_dict:
        out_width += img_pair_dict[key]["cur_width"]
        out_width += 2 * result_bbox_width
        out_width += img_col_space
    out_width = max([out_width, src_w])
    # put src lab
    if trans_text is not None and trans_text != "":
        font_size, area_height = adjuct_font_and_height(trans_text, out_width, 30)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        
        trans_img = drawText(trans_text, area_height, out_width, font)
        # pdb.set_trace()
        new_src_img = 255*np.ones((src_h + area_height, out_width, 3), dtype=np.uint8)
        new_src_img[:src_h, :src_w, :] = src_img
        new_src_img[src_h:, :, :] = trans_img
        #src_img = np.vstack([src_img, trans_img])
        src_img = new_src_img
        src_h, src_w, src_c = src_img.shape
    
    out_height = labs_height + recs_height + src_h + img_line_space*2 + result_bbox_width * 4
    outImg = 255*np.ones((out_height, out_width, 3), dtype=np.uint8)
    # pdb.set_trace()
    #put src img
    src_pos_y = labs_height +img_line_space+ result_bbox_width * 2
    outImg[src_pos_y:src_pos_y+src_h, :src_w, :] = src_img
    #put recs
    cur_pos_x = 0
    lab_pos_y = 0
    rec_pos_y = src_pos_y + src_h + img_line_space

    for key in img_pair_dict:
        lab_img = img_pair_dict[key]["lab_img"]
        lab_img_h, lab_img_w, _ = lab_img.shape
        container_w = img_pair_dict[key]["cur_width"]
        container_h = labs_height
        if rec_rep_dict[key]["res_cmp"] == 0: #correct
            outImg[lab_pos_y:lab_pos_y+container_h+2*result_bbox_width, cur_pos_x:cur_pos_x+container_w+2*result_bbox_width, :] = 0
        else:
            outImg[lab_pos_y:lab_pos_y+container_h+2*result_bbox_width, cur_pos_x:cur_pos_x+container_w+2*result_bbox_width, :2] = 0
        lab_img_expand = 255*np.ones((container_h, container_w, 3), dtype=np.uint8)
        t_x = (container_w - lab_img_w) // 2
        t_y = (container_h - lab_img_h) // 2
        lab_img_expand[t_y:t_y+lab_img_h, t_x:t_x+lab_img_w, :] = lab_img
        outImg[lab_pos_y+result_bbox_width:lab_pos_y+result_bbox_width+container_h, cur_pos_x+result_bbox_width:cur_pos_x+result_bbox_width+container_w, :] = lab_img_expand

        rec_img = img_pair_dict[key]["rec_img"]
        rec_img_h, rec_img_w, _ = rec_img.shape
        container_w = img_pair_dict[key]["cur_width"]
        container_h = recs_height
        if rec_rep_dict[key]["res_cmp"] == 0: #correct
            outImg[rec_pos_y:rec_pos_y+container_h+2*result_bbox_width, cur_pos_x:cur_pos_x+container_w+2*result_bbox_width, :] = 0
        else:
            outImg[rec_pos_y:rec_pos_y+container_h+2*result_bbox_width, cur_pos_x:cur_pos_x+container_w+2*result_bbox_width, :2] = 0
        rec_img_expand = 255*np.ones((container_h, container_w, 3), dtype=np.uint8)
        t_x = (container_w - rec_img_w) // 2
        t_y = (container_h - rec_img_h) // 2
        rec_img_expand[t_y:t_y+rec_img_h, t_x:t_x+rec_img_w, :] = rec_img
        outImg[rec_pos_y+result_bbox_width:rec_pos_y+result_bbox_width+container_h, cur_pos_x+result_bbox_width:cur_pos_x+result_bbox_width+container_w, :] = rec_img_expand

        cur_pos_x += (container_w + img_col_space + result_bbox_width * 2)
    return outImg

def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", type=str, default="")
    args = parser.parse_args()
    main(args)
