# Medical Hypernymy Detection Dataset

This repository provides two corpora which contain hypernymy pairs of medical terms, both in English and Chinese. The corpora also contain negative instances, and have been splited into training sets, test sets and validation sets.

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

## English Corpus
We extract terms which are labeled as "clinical finding" in SNOMED CT, and their children to construct positive hypernymy instances. We also take hyponymy, synonymy and unrelated pairs as negative instances. 

## Chinese Corpus
We select six Chinese healthcare websites, and extract hypernymy and synonymy relations between symptoms from semi-structured and unstructured data on the detail pages. We set hypernymy symptom pairs as positive instances, hyponymy, synonymy and unrelated symptom pairs as negative instances.

### Six Chinese Healthcare Websites
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
