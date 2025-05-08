import textdistance
import re
from itertools import combinations

def extract_docstring(prompt):
    """从prompt中提取三引号内的文档字符串"""
    match = re.search(r'\"\"\"(.*?)\"\"\"', prompt, re.DOTALL)
    return match.group(1).strip() if match else ""
    
def preprocess(text):
    """统一预处理规则"""
    # 移除代码示例中的具体数值
    text = re.sub(r'\d+', '#', text)
    # 统一转小写并去除标点
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def hybrid_similarity(text1, text2):
    """混合相似度计算"""
    # 预处理文本
    t1, t2 = preprocess(text1), preprocess(text2)
    
    # 算法组合
    return {
        'jaccard': textdistance.jaccard.normalized_similarity(t1, t2),
        'cosine': textdistance.cosine(t1, t2),
        'sorensen': textdistance.sorensen(t1, t2),
        'normalized_levenshtein': textdistance.levenshtein.normalized_similarity(t1, t2)
    }

def get_similarity_score(text1, text2):
    scores = hybrid_similarity(text1, text2)
    return max(scores.values())

# 测试示例
if __name__ == "__main__":
    # 执行该文件以验证是否可执行
    text1 = """From a list of integers, remove all elements that occur more than once.\n    Keep order of elements left the same as in the input.\n    >>> remove_duplicates([1, 2, 3, 2, 4])\n    [1, 3, 4]\n"""  # 原始基准文本
    text2 = """Remove duplicate integers from a list, returning only the integers that appear once.\n    \n    The function creates a count dictionary to track the occurrences of each number and filters the list to return only those that occur exactly once.\n\n    Examples:\n    remove_duplicates([1, 2, 2, 3, 4, 4, 5]) == [1, 3, 5]\n    remove_duplicates([10, 10, 10, 20, 30, 30]) == [20]\n    remove_duplicates([5, 5, 5]) == []\n"""  # 生成文本
    
    scores = hybrid_similarity(text1, text2)
    print(f"综合相似度：{max(scores.values()):.2f}")