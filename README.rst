|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Uncertainty Estimation Methods for Countering Attacks on Machine-Generated Text Detectors
    :Тип научной работы: M1P
    :Автор: Леванов Валерий Дмитриевич
    :Научный руководитель: к. ф-м н. Грабовой Андрей Валериевич,
    :Научный консультант: Вознюк Анастасия Евгеньевна 
Abstract
========

	This study investigates the application of uncertainty estimation methods to enhance the quality of machine-generated text detectors when processing data containing attacks such as homoglyphs, paraphrasing, and noise injection. These attacks are not only used to bypass detection but also serve to test the robustness of detectors. We examine the hypothesis that uncertainty estimation methods can provide a more resilient approach, eliminating the need for continuous retraining across various attack types. We propose a method combining uncertainty estimation with classifiers based on hidden representations of language models. Experiments on the M4GT and RAID datasets demonstrate competitive accuracy (ROC-AUC 0.8977) with significantly lower computational costs compared to fine-tuning large language models (LLMs).
