from peewee import IntegerField, TextField, AutoField, Model, SqliteDatabase, CharField

db = SqliteDatabase('chant_oracle_new.db')


class BaseModel(Model):
    class Meta:
        database = db


class ChantOracleBoneItems(BaseModel):
    """
    甲片条目表，每条数据代表甲片中的一个条目，(著录号-条号) 唯一对应一条数据
    """
    # 自增 id 域，主键
    id = AutoField()
    # 著录号（甲片的唯一标识），最大长度 511 字符, 原 book_name
    chant_published_collection_number = \
        CharField(null=False, max_length=511, column_name='chant_published_collection_number')
    # 汉达条号（正整数，甲片下某个甲骨文句子的序号），原 row_order
    chant_notation_number = IntegerField(null=False, column_name='chant_notation_number')
    # 汉达释文，甲骨文句子（繁体汉字），带 font 标签，原 modern_text
    chant_transcription_text = TextField(null=False, column_name='chant_transcription_text')
    # 汉达文库字体分类，最大长度 7，原 category
    chant_calligraphy = CharField(null=False, max_length=7, column_name='chant_calligraphy')
    # 源数据在汉达文库中的 url 后缀，最大长度 511 字符，原 url
    chant_url = CharField(null=False, max_length=511, column_name='chant_url')
    # 包含字形的列表，以 '\t' 分隔的字符串，每个元素都是字形表 CharShape.id
    characters = TextField(null=False, column_name='characters')
    # 甲片图的路径，最大长度 511 字符，原 l_bone_img
    chant_processed_rubbing = CharField(null=False, max_length=511, column_name='chant_processed_rubbing')
    # 汉字排布图的路径，最大长度 511 字符，原 r_bone_img
    chant_mapped_character_image = CharField(null=False, max_length=511, column_name='chant_mapped_character_image')
    # 数据来源，新增列，如有多个来源，请用 \t 分割，默认值为 chant.org
    data_sources = TextField(null=False, default='chant.org', column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 著录号-条号 联合唯一索引
        indexes = (
            (('chant_published_collection_number', 'chant_notation_number'), True),
        )


class Character(BaseModel):
    """
    标准字表，一个标准字（单字+合文）包含多个字形，(编码-字体) 不一定唯一对应一个标准字！同一个汉字可能对应多个 Character（编码不同）!
    """
    # 自增 id 域，主键
    id = AutoField()
    # 摹本中的编号，索引字段，为 -1 表示只在汉达而不在摹本中的字，原 char_index
    wzb_character_number = IntegerField(null=False, index=True, column_name='wzb_character_number')
    # 汉达字体标签 name，原 font
    chant_font_label = CharField(null=False, max_length=7, column_name='chant_font_label')
    # 《字表目录》第1列“字形”图片路径，新增列，默认为空
    standard_inscription = CharField(null=False, default="", max_length=511, column_name='standard_inscription')
    # 《字表目录》第2列“隶定”现代汉字，utf-8 编码，原 char_byte
    standard_liding_character = CharField(null=False, max_length=7,
                                          column_name='standard_liding_character')
    # 一级字头，新增列，默认为空
    first_order_standard_character = CharField(null=False, default="", max_length=511,
                                               column_name='first_order_standard_character')
    # 二级字头，新增列，默认为空
    second_order_standard_character = CharField(null=False, default="", max_length=511,
                                                column_name='second_order_standard_character')
    # 《字表目录》第4列“页码”，原 page_number，默认为 -1
    wzb_page_number = IntegerField(null=False, default=-1, column_name='wzb_page_number')
    # 部首编号，未指定时为 -1
    wzb_radical_number = IntegerField(null=False, default=-1, column_name='wzb_radical_number')
    # 部首，新增列，默认为空
    wzb_radical = CharField(null=False, default="", max_length=7, column_name='wzb_radical')
    # 拼音，新增列，默认为空
    wzb_spelling = CharField(null=False, default="", max_length=511, column_name='wzb_spelling')
    # 笔画数，新增列，默认为 -1
    wzb_stroke_count = IntegerField(null=False, default=-1, column_name='wzb_stroke_count')
    # 摹本中该字在目录中位于目录的哪一页，-1 表示未处理或非摹本中的字
    wzb_table_page = IntegerField(null=False, default=-1, column_name='wzb_table_page')
    # 摹本中该字在目录中位于目录的哪一行，-1 表示未处理或非摹本中的字
    wzb_table_row = IntegerField(null=False, default=-1, column_name='wzb_table_column')
    # 摹本中该字在目录中位于目录的哪一栏，0 表示左边栏，1 表示右边栏，-1 表示未处理或非摹本中的字
    # TODO: 目前针对 1-4 栏的问题，如果需要将数据进行精准识别和清晰后再放入数据库
    wzb_table_col = IntegerField(null=False, default=-1, column_name='wzb_table_col')
    # 数据来源，新增列，如有多个来源，请用 \t 分割
    data_sources = TextField(null=False, column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 编码-字体 联合唯一索引
        indexes = (
            (('standard_liding_character', 'chant_font_label'), False),
        )


class CharFace(BaseModel):
    """
    字形表，对应汉达文库/李老师摹本中的每一个字形
    """
    # 自增 id 域，主键
    id = AutoField()
    # 根据 (著录号 - 无字体汉字) 进行匹配
    # 0-只在汉达中，不在摹本中；1-只在摹本中，不在汉达中；2-同时存在，数据可以对上，一一对应；3-同时存在，数据可对上，多于1条
    match_case = IntegerField(null=False, index=True, column_name='match_case')
    # 属于哪一个标准字 Character.id，索引字段，原 char_belong，可以由此索引到一级和二级字头
    standard_character_id = IntegerField(null=False, index=True, column_name='standard_character_id')
    # 属于哪一个标准字，包含了汉达/摹本的现代汉字标准字，默认为空
    standard_character = CharField(null=False, default="", max_length=7, column_name='standard_character')
    # 在汉达文库甲片图中的坐标信息，可能为空（match_case == 1），原 coords
    chant_coordinates = CharField(null=False, max_length=63, column_name='chant_coordinates')
    # 原形，即汉达文库中带背景的噪声图片路径，最大长度 511 字符，可能为空（match_case == 1），原 noise_image
    chant_authentic_face = CharField(null=False, max_length=511, column_name='chant_authentic_face')
    # 汉达条号（正整数，甲片下某个甲骨文句子的序号），原 row_order
    chant_notation_number = IntegerField(null=False, default=-1, column_name='chant_notation_number')
    # 汉达中文字图片的编号，（著录号+条号+文字图片）
    chant_face_index = IntegerField(null=False, default=-1, column_name='chant_face_index')
    # 摹写字形图片路径，最大长度 511 字符，可能为空（match_case == 0），原 shape_image
    wzb_handcopy_face = CharField(null=False, max_length=511, column_name='wzb_handcopy_face')
    # 所属的著录号，最大长度 511 字符，match_case == 0/2-取汉达著录号表示，1-取摹本著录号表示，原 book_name
    published_collection_number = CharField(null=False, max_length=511, column_name='published_collection_number')
    # 李老师摹本字体分类，最大长度 7，可能为空（match_case == 0），missing 表示找不到有效 ocr 编码，原 category
    wzb_calligraphy = CharField(null=False, max_length=7, column_name='wzb_calligraphy')
    # 页码号，可能为 -1（match_case == 0），原 page_number
    wzb_page_number = IntegerField(null=False, column_name='wzb_page_number')
    # 第几行，可能为 -1（match_case == 0）
    wzb_row_number = IntegerField(null=False, column_name='wzb_row_number')
    # 第几列，可能为 -1（match_case == 0）
    wzb_col_number = IntegerField(null=False, column_name='wzb_col_number')
    # 数据来源，新增列，如有多个来源，请用 \t 分割
    data_sources = TextField(null=False, column_name='data_sources')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 汉达字形图片的联合唯一索引，著录号-条号-编号
        indexes = (
            (('published_collection_number', 'chant_notation_number', 'chant_face_index'), False),
        )


def init_db():
    db.connection()


def get_files():
    '''
    Return generator of pairs of corresponding original noisy and transcription
    file.
    '''
    init_db()
    # for char_shape in CharShape.select().where(CharShape.match_case == 2):
        # yield char_shape.noise_image, char_shape.shape_image
    char_faces = CharFace.select().where(CharFace.match_case == 2)
    for char_face in char_faces:
        # print(char_face.chant_authentic_face, char_face.wzb_handcopy_face)
        yield char_face.chant_authentic_face, char_face.wzb_handcopy_face


if __name__ == '__main__':
    files = get_files()
    count = 0
    for noisy, transcript in files:
        # print(noisy, transcript)
        count += 1
    print(count)
