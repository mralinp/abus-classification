"""Fully automatic classification of ABUS according to BI-RADS using a deep CNN (Eur. Radiol. 2022)."""
from abus_classification.papers.hejduk2022.model import HejdukCNN
from abus_classification.papers.hejduk2022.pipeline import main, train_model
