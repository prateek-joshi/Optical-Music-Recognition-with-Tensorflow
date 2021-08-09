import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                         root.find('size').find('width').text,
                         root.find('size').find('height').text,
                         member[0].text,
                         member.find("bndbox").find('xmin').text,
                         member.find("bndbox").find('ymin').text,
                         member.find("bndbox").find('xmax').text,
                         member.find("bndbox").find('ymax').text
                        )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for directory in ['train','test']:
        image_path = f'Tensorflow/workspace/images/{directory}'
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(f'Tensorflow/workspace/annotations/{directory}_labels.csv', index=None)
        print(f'Successfully converted xml to csv - {directory}.csv')


main()