import os
import sys
import json
import numpy as np
#import openai
#from openai_utils import OPENAI_API_KEY
#openai.api_key = OPENAI_API_KEY
import random
from tqdm import tqdm
from influence_on_human_ratings import load_human_ratings_data, adjust_impath, BASE_DIR
from ontology_questions import parse_semantic_graph, get_log_filename
from ontology_eval import SKIP_FILENAMES
from winoground_inference_with_LLM_fusion import ask_deepseek


DATA_STRIDE = 24
QUESTIONS_PREFIX = os.path.join(BASE_DIR, 'genaibench_questions_from_graphs')


#return semantic_graph
def extract_semantic_graph(log_filename):
    lines = []
    is_reading = False
    success = False
    f = open(log_filename, 'r')
    for line in f:
        s = line.rstrip('\n')
        if len(s) == 0:
            continue

        if is_reading:
            if s == 'RUNNING OBJECT DETECTOR...':
                success = True
                break
            elif s == 'GIVE UP ON GRAPH':
                break
            else:
                lines.append(s)
        elif s == 'SEMANTIC GRAPH:':
            is_reading = True

    f.close()
    if not success:
        return None

    semantic_graph = json.loads('\n'.join(lines))
    return semantic_graph


def create_object_texts_one(my_object, LLM_cache, is_plural, include_attributes):
    if include_attributes and len(my_object['attributes']) > 0:
        attr_str = ', '.join(['"%s"'%(attribute['attribute_name'].lower()) for attribute in my_object['attributes']])
        if is_plural:
            content = 'You are an expert in grammar and writing. Please give the grammatically correct phrase for the object "%s", in its plural form, with attribute(s) %s, being as faithful to the original text as possible. Put the answer by itself in the last line of your response. Do not give me a complete sentence, put the answer by itself.'%(my_object['name'].lower(), attr_str)
        else:
            content = 'You are an expert in grammar and writing. Please give the grammatically correct phrase for the object "%s" with attribute(s) %s, being as faithful to the original text as possible. Put the answer by itself in the last line of your response. Do not give me a complete sentence, put the answer by itself.'%(my_object['name'].lower(), attr_str)
    else:
        if is_plural:
            content = 'You are an expert in grammar and writing. Please give the plural form of "%s", being as faithful to the original text as possible. Put the answer by itself in the last line of your response. Do not give me a complete sentence, put the answer by itself.'%(my_object['name'])
        else:
            return my_object['name']

    if content in LLM_cache:
        reply = LLM_cache[content]
    else:
        reply = ask_deepseek(content)
        LLM_cache[content] = reply

    reply = reply.strip().split('\n')[-1].strip().strip('"').strip().rstrip('.')
    return reply


#return dictionary mapping from object_id to (singular_object_text, plural_object_text, singular_object_text_with_attributes, plural_object_text_with_attributes)
def create_object_texts(semantic_graph):
    if len(semantic_graph['objects']) == 0:
        return {}

    object_texts = {}
    #this should keep us from having to worry about whether we're processing duplicate objects
    #if we do, then it'll be a duplicate query to the LLM cache, so the extraneous LLM work will be avoided
    LLM_cache = {}
    for my_object in semantic_graph['objects']:
        kk = {}
        for is_plural in [False, True]:
            for include_attributes in [False, True]:
                s = create_object_texts_one(my_object, LLM_cache, is_plural, include_attributes)
                kk[['singular', 'plural'][is_plural] + '_object_text' + ['', '_with_attributes'][include_attributes]] = s

        object_texts[my_object['object_id']] = kk

    return object_texts


#map from (singular_text, plural_text) to count
def make_text_counter(object_list, object_texts, include_attributes):
    text_counter = {}
    attr_str = {False : '', True : '_with_attributes'}[include_attributes]
    for my_object in object_list:
        k = my_object['object_id']
        is_group = {'Yes' : True, 'No' : False}[my_object['is_group']]
        singular_text = object_texts[k]['singular_object_text' + attr_str]
        plural_text = object_texts[k]['plural_object_text' + attr_str]
        kk = (singular_text, plural_text)
        if kk not in text_counter:
            text_counter[kk] = 0

        if is_group:
            text_counter[kk] = np.inf
        else:
            text_counter[kk] += 1

    return text_counter


def make_object_dict(semantic_graph):
    objects = {}
    for my_object in semantic_graph['objects']:
        k = my_object['object_id']
        assert(k not in objects)
        objects[k] = my_object

    return objects


#if there are multiple questions with the same text, only keep the first one in the list
def deduplicate_questions(questions):
    already = set([])
    filtered_questions = []
    for question in questions:
        if question[0] in already:
            continue

        already.add(question[0])
        filtered_questions.append(question)

    return filtered_questions


def generate_presence_and_count_questions_helper(semantic_graph, object_texts, include_attributes):
    questions = []
    attr_str = {False : '', True : '_with_attributes'}[include_attributes]

    text_counter = make_text_counter(semantic_graph['objects'], object_texts, include_attributes)

    #at least one
    for kk in sorted(text_counter.keys()):
        questions.append(('Is there at least one %s in the image?'%(kk[0]), ('yes', 'no'), 'yes', 'at_least_one' + attr_str))

    #at least two
    for kk in sorted(text_counter.keys()):
        if text_counter[kk] >= 2:
            questions.append(('Are there at least two %s in the image?'%(kk[1]), ('yes', 'no'), 'yes', 'at_least_two' + attr_str))

    #exact count
    for kk in sorted(text_counter.keys()):
        if np.isinf(text_counter[kk]):
            continue

        if text_counter[kk] == 1:
            questions.append(('Is there exactly one %s in the image?'%(kk[0]), ('yes', 'no'), 'yes', 'exact_count' + attr_str))
        else:
            questions.append(('Are there exactly %d %s in the image?'%(text_counter[kk], kk[1]), ('yes', 'no'), 'yes', 'exact_count' + attr_str))

    return questions


def generate_presence_and_count_questions(semantic_graph, object_texts):
    questions = []
    if len(semantic_graph['objects']) == 0:
        return []

    questions.extend(generate_presence_and_count_questions_helper(semantic_graph, object_texts, False))
    questions.extend(generate_presence_and_count_questions_helper(semantic_graph, object_texts, True))
    return questions


def generate_single_relationship_questions_one(objects, relationship, object_texts, include_attributes):
    questions = []
    attr_str = {False : '', True : '_with_attributes'}[include_attributes]

    text_counter_subj = make_text_counter([objects[k] for k in relationship['subject_ids']], object_texts, include_attributes)
    text_counter_obj = make_text_counter([objects[k] for k in relationship['object_ids']], object_texts, include_attributes)

    for kk_subj in sorted(text_counter_subj.keys()):
        for kk_obj in sorted(text_counter_obj.keys()):
            subj_is_plural = (text_counter_subj[kk_subj] > 1)
            obj_is_plural = (text_counter_obj[kk_obj] > 1)
            subj_text = kk_subj[subj_is_plural]
            obj_text = kk_obj[obj_is_plural]
            words = [['Is', 'Are'][subj_is_plural], 'there']
            if not subj_is_plural:
                words.append(['a', 'an'][subj_text[0].lower() in 'aeiou'])

            words.append(subj_text)
            words.extend(['that', ['is', 'are'][subj_is_plural]])
            words.append(relationship['predicate'])
            if obj_is_plural:
                words.append('the')
            else:
                words.append(['a', 'an'][obj_text[0].lower() in 'aeiou'])

            if obj_text == subj_text:
                words.append('other')

            words.append(obj_text)
            question = (' '.join(words) + '?', ('yes', 'no'), 'yes', 'single_relationship' + attr_str)
            assert(question not in questions)
            questions.append(question)

    return questions


#"[Is/Are] there [a/an] <NODE_I> that [is/are] <PREDICATE> the [other] <NODE_J>?"
def generate_single_relationship_questions(semantic_graph, object_texts):
    questions = []
    if len(semantic_graph['objects']) == 0:
        return []
    
    objects = make_object_dict(semantic_graph)
    for relationship in semantic_graph['relationships']:
        questions.extend(generate_single_relationship_questions_one(objects, relationship, object_texts, False))

    for relationship in semantic_graph['relationships']:
        questions.extend(generate_single_relationship_questions_one(objects, relationship, object_texts, True))

    return questions


#you can give it any two relationships, it'll return empty if they have nothing in common
def generate_double_relationship_questions_ijik_one(objects, relationship_ij, relationship_ik, object_texts, include_attributes):
    questions = []
    attr_str = {False : '', True : '_with_attributes'}[include_attributes]

    subj_ids = sorted(set(relationship_ij['subject_ids']) & set(relationship_ik['subject_ids']))
    obj_ij_ids = sorted(set(relationship_ij['object_ids']) - set(relationship_ik['object_ids']))
    obj_ik_ids = sorted(set(relationship_ik['object_ids']) - set(relationship_ij['object_ids']))
    text_counter_subj = make_text_counter([objects[k] for k in subj_ids], object_texts, include_attributes)
    text_counter_ij_obj = make_text_counter([objects[k] for k in obj_ij_ids], object_texts, include_attributes)
    text_counter_ik_obj = make_text_counter([objects[k] for k in obj_ik_ids], object_texts, include_attributes)

    for kk_subj in sorted(text_counter_subj.keys()):
        for kk_obj_ij in sorted(text_counter_ij_obj.keys()):
            for kk_obj_ik in sorted(text_counter_ik_obj.keys()):
                subj_is_plural = (text_counter_subj[kk_subj] > 1)
                obj_ij_is_plural = (text_counter_ij_obj[kk_obj_ij] > 1)
                obj_ik_is_plural = (text_counter_ik_obj[kk_obj_ik] > 1)
                subj_text = kk_subj[subj_is_plural]
                obj_ij_text = kk_obj_ij[obj_ij_is_plural]
                obj_ik_text = kk_obj_ik[obj_ik_is_plural]
                ij_same = int(subj_text == obj_ij_text)
                ik_same = int(subj_text == obj_ik_text)
                jk_same = int(obj_ij_text == obj_ik_text)
                words = [['Is', 'Are'][subj_is_plural], 'there']
                if not subj_is_plural:
                    words.append(['a', 'an'][subj_text[0].lower() in 'aeiou'])

                words.append(subj_text)
                subj_isare = ['is', 'are'][subj_is_plural]
                words.extend(['that', subj_isare])
                words.append(relationship_ij['predicate'])
                if ij_same:
                    words.append(['another', 'the other'][obj_ij_is_plural])
                elif obj_ij_is_plural:
                    words.append('the')
                else:
                    words.append(['a', 'an'][obj_ij_text[0].lower() in 'aeiou'])

                words.append(obj_ij_text)
                words.extend(['and', 'also', subj_isare])
                words.append(relationship_ik['predicate'])
                if ik_same + jk_same == 0: #no othering needed
                    if obj_ik_is_plural:
                        words.append('the')
                    else:
                        words.append(['a', 'an'][obj_ik_text[0].lower() in 'aeiou'])
                elif ij_same == 0: #othering needed, but this is only the first time
                    words.append(['another', 'the other'][obj_ik_is_plural])
                else: #othering needed, and this is not the first time
                    words.append('some other')

                words.append(obj_ik_text)
                question = (' '.join(words) + '?', ('yes', 'no'), 'yes', 'double_relationship_ijik' + attr_str)
                assert(question not in questions)
                questions.append(question)

    return questions


#"[Is/Are] there [a/an] <NODE_I> that [is/are] <PREDICATE_IJ> [a/an/the/another/the other] <NODE_J> and also <PREDICATE_IK> [a/the/another/the other/some other] <NODE_K>?"
def generate_double_relationship_questions_ijik(semantic_graph, object_texts):
    questions = []
    if len(semantic_graph['objects']) == 0:
        return []

    objects = make_object_dict(semantic_graph)
    for a in range(len(semantic_graph['relationships']) - 1):
        for b in range(a + 1, len(semantic_graph['relationships'])):
            relationship_ij = semantic_graph['relationships'][a]
            relationship_ik = semantic_graph['relationships'][b]
            questions.extend(generate_double_relationship_questions_ijik_one(objects,relationship_ij,relationship_ik,object_texts,False))

    for a in range(len(semantic_graph['relationships']) - 1):
        for b in range(a + 1, len(semantic_graph['relationships'])):
            relationship_ij = semantic_graph['relationships'][a]
            relationship_ik = semantic_graph['relationships'][b]
            questions.extend(generate_double_relationship_questions_ijik_one(objects,relationship_ij,relationship_ik,object_texts,True))

    return questions


def generate_scene_and_absence_questions(semantic_graph):
    questions = []
    scene = semantic_graph['scene']
    scene_description = scene['description']
    absent_object_names = scene['absent_object_names']
    if scene_description.lower() != 'n/a':
        questions.append(('Could the scene or style of this image be described as "%s"?'%(scene_description), ('yes', 'no'), 'yes', 'scene_description'))

    for name in absent_object_names:
        questions.append(('Is there any %s in the image?'%(name), ('yes', 'no'), 'no', 'absent_object'))

    return questions


def get_question_log_filename(log_filename):
    return os.path.join(BASE_DIR, 'genaibench_questions_from_graphs', os.path.basename(log_filename))


#return list of questions
#each item is (question, correct_answer, question_type)
def generate_questions_from_semantic_graph_one(semantic_graph):
    questions = []
    object_texts = create_object_texts(semantic_graph)
    questions.extend(generate_presence_and_count_questions(semantic_graph, object_texts))
    questions.extend(generate_single_relationship_questions(semantic_graph, object_texts))
    questions.extend(generate_double_relationship_questions_ijik(semantic_graph, object_texts))
    questions.extend(generate_scene_and_absence_questions(semantic_graph))
    questions = deduplicate_questions(questions)
    questions = [{'question' : q[0], 'choices' : list(q[1]), 'answer' : {'yes':1,'no':0}[q[2]], 'type' : q[3]} for q in questions]
    return questions


def save_questions_data(genaibench_questions_data, questions_filename):
    with open(questions_filename, 'w') as f:
        json.dump(genaibench_questions_data, f)


def generate_questions_from_semantic_graph(offset):
    offset = int(offset)

    questions_filename = QUESTIONS_PREFIX + '_%d.json'%(offset)
    genaibench_questions_data = {}
    if os.path.exists(questions_filename):
        with open(questions_filename, 'r') as f:
            genaibench_questions_data = json.load(f)

        print('oh goody, already have %d processed!'%(len(genaibench_questions_data)))

    d_image = load_human_ratings_data()
    for t, k in tqdm(enumerate(sorted(d_image.keys())[offset::DATA_STRIDE])):
        for generator in sorted(d_image[k]['images'].keys()):
            if str((k, generator)) in genaibench_questions_data:
                print('%s already computed, skip'%(str((k, generator))))
                continue

            my_input = d_image[k]['prompt']
            print(my_input)
            image_filename = adjust_impath(d_image[k]['images'][generator]['filename'])
            log_filename = get_log_filename(my_input, image_filename)
            if not os.path.exists(log_filename) or os.path.basename(log_filename) in SKIP_FILENAMES:
                print('!!')
                continue

            semantic_graph = extract_semantic_graph(log_filename)
            if semantic_graph is None:
                print('!')
                continue

            questions = generate_questions_from_semantic_graph_one(semantic_graph)
            print(questions)
            genaibench_questions_data[str((k, generator))] = questions
            if t % 5 == 0 and t > 0:
                save_questions_data(genaibench_questions_data, questions_filename)

    save_questions_data(genaibench_questions_data, questions_filename)


def usage():
    print('Usage: python generate_questions_from_semantic_graph.py <offset>')


if __name__ == '__main__':
    generate_questions_from_semantic_graph(*(sys.argv[1:]))
