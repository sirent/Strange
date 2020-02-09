post = ["@fandyaulia87 😤lo yg gw bikin busuk loh. Loh gerombolan lgbt yaa😂", "@jokowi dia siapa", "aku siapa"]
labels = ['hate','not_hate','not_hate']

def extract_username(comment):
    # uname = re.compile(r'@([^\s:]+)')
    s_comment = str(comment)

    uname = re.compile(r'@([A-Za-z0-9_]+)')
    comments = re.sub('(@[^\s]+)', 'N_Target', s_comment)
    target = uname.findall(s_comment)

    commentss = []
    commentss.append(comments)
    print(commentss)
    print(labels)
    print(target)
    new_data_train = pd.DataFrame(list(zip(post, labels, target)),columns=['Comment', 'Label', 'Target'])
    print(new_data_train)
