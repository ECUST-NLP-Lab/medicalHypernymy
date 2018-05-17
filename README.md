# An Attention-based Bi-GRU-CapsNet Model for Hypernymy Detection between Compound Entities

This repository contains the experiments done in the work <a href="https://arxiv.org/abs/1805.04827">An Attention-based Bi-GRU-CapsNet Model for Hypernymy Detection between Compound Entities</a> by Qi Wang, Chenming Xu, Tong Ruan, Yangming Zhou, Daqi Gao and Ping He.
Named entities which composed of multiple continuous words frequently occur in knowledge graphs for biomedical sciences. These entities are usually composable and extensible. Typical examples are names of symptoms and diseases. To distinguish these entities from general entities, we name them compound entities.
Hypernymy detection is useful for natural language processing (NLP) tasks such as taxonomy creation, ontology extension, textual entailment recognition, sentence similarity estimation and text generation. However, existing methods for hypernymy detection deal with the case where an entity only includes a word. In this work, we present a novel attention-based Bi-GRU-CapsNet model to detect hypernymy relationship between compound entities.
Our model integrates several important components. English words or Chinese characters in compound entities are fed into Bidirectional Recurrent Units (Bi-GRUs) to avoid the Out-Of-Vocabulary (OOV) problem. An attention mechanism is then designed to focus on the differences between two compound entities. Since there are different cases in hypernymy relationship between compound entities, Capsule Network (CapsNet) is finally employed to decide whether the hypernymy relationship exists or not. Experimental results demonstrate the advantages of the proposed model over the state-of-the-art methods both on English and Chinese corpora of symptom and disease pairs.
This repository provides two corpora which contain hypernymy pairs of symptoms and diseases in English and Chinese. We will release the source code of this work after publication.

##Datasets
This repository provides two corpora which contain hypernymy pairs of clinical findings in English and Chinese. The corpora also contain negative instances, and have been splited into training sets, test sets and validation sets.
<table>
  <tr>
    <th  align="center" colspan="2">Dataset</th>
    <th  align="center">Positive</th>
    <th  align="center">Negative</th>
    <th  align="center">All</th>
  </tr>
  <tr>
    <td  align="center" rowspan="3">English</td>
    <td  align="center">Train</td>
    <td  align="center">27,872</td>
    <td  align="center">27,872</td>
    <td  align="center">55,744</td>
  </tr>
  <tr>
    <td  align="center">Test</td>
    <td  align="center">9,954</td>
    <td  align="center">9,954</td>
    <td  align="center">19,908</td>
  </tr>
  <tr>
    <td  align="center">Val</td>
    <td  align="center">1,991</td>
    <td  align="center">1,991</td>
    <td  align="center">3,982</td>
  </tr>
  <tr>
    <td  align="center" rowspan="3">Chinese</td>
    <td  align="center">Train</td>
    <td  align="center">8,960</td>
    <td  align="center">8,960</td>
    <td  align="center">17,920</td>
  </tr>
  <tr>
    <td  align="center">Test</td>
    <td  align="center">3,200</td>
    <td  align="center">3,200</td>
    <td  align="center">6,400</td>
  </tr>
  <tr>
    <td  align="center">Val</td>
    <td  align="center">640</td>
    <td  align="center">640</td>
    <td  align="center">1,280</td>
  </tr>
</table>

### English Corpus
We extract terms which are labeled as "clinical finding" in SNOMED CT, and their children to construct positive hypernymy instances. We also take hyponymy and unrelated pairs as negative instances. 

### Chinese Corpus
We select six Chinese healthcare websites, and extract hypernymy and synonymy relations between symptoms from semi-structured and unstructured data on the detail pages. We set hypernymy symptom pairs as positive instances, hyponymy, synonymy and unrelated symptom pairs as negative instances.

#### Six Chinese Healthcare Websites
<table>
  <tr>
    <th  align="center">Website</th>
    <th  align="center">URL</th>
  </tr>
  <tr>
    <td  align="center">XYWY</td>
    <td  align="center"><a href="http://www.xywy.com/">http://www.xywy.com/</a></td>
  </tr>
  <tr>
    <td  align="center">120ask</td>
    <td  align="center"><a href="http://www.120ask.com/">http://www.120ask.com/</a></td>
  </tr>
  <tr>
    <td  align="center">39Health<br></td>
    <td  align="center"><a href="http://www.39.net/">http://www.39.net/</a></td>
  </tr>
  <tr>
    <td  align="center">99Health</td>
    <td  align="center"><a href="http://www.99.com.cn/">http://www.99.com.cn/</a></td>
  </tr>
  <tr>
    <td  align="center">Familydoctor</td>
    <td  align="center"><a href="http://www.familydoctor.com.cn/">http://www.familydoctor.com.cn/</a></td>
  </tr>
  <tr>
    <td  align="center">Fh21</td>
    <td  align="center"><a href="http://www.fh21.com.cn/">http://www.fh21.com.cn/</a></td>
  </tr>
</table>

##Source Code
We will release the source code of this work after publication.