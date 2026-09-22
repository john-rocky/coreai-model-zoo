#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "torch==2.9.0", "transformers==5.16.1", "safetensors==0.8.0",
#   "huggingface_hub==1.32.0", "numpy==2.3.5", "tokenizers==0.23.2",
# ]
# ///
"""Author OpenJet CPU/fp32 oracle, or validated replay of an accepted fixture.

The 48 deterministic requests are embedded below. --replay-fixtures preserves
accepted bytes and checks them through the vendored author's compiler, without
loading model weights or re-running numerical inference. Without that option,
the full-depth high API and 16-layer low API execute for every request.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import argparse
import copy
import re
import shutil
from types import SimpleNamespace

DEFAULT_HF_ID = "apus-ailab/APUS-OpenJev-v1-4B"
DEFAULT_REVISION = "65797c526c27c4d24f564333779162cd4a64328e"
WORK = Path.cwd()
REQUESTS = json.loads(r'''[
  {
    "request": {
      "id": "choice16_01",
      "group_id": "choice16",
      "state": "{\"goal\": \"find the lunar archive schedule\", \"history\": \"The page has just opened; no search has been submitted.\", \"loading\": false, \"page\": \"document search\", \"search_field\": \"empty\", \"visible_controls\": [\"search field\", \"Search button\"]}",
      "instructions": "Choose the next browser action that advances the stated goal. The search field must contain the goal query before a search can be submitted.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "go_back",
          "description": "Return to the previous page without changing any data."
        },
        {
          "id": "next_page",
          "description": "Open the next page of the paginated results."
        },
        {
          "id": "close_banner",
          "description": "Dismiss the visible informational banner with its close button."
        },
        {
          "id": "save_draft",
          "description": "Save the completed form as a draft without submitting it."
        },
        {
          "id": "submit_form",
          "description": "Submit the completed form once all required fields are valid."
        },
        {
          "id": "fix_required",
          "description": "Fill the required field that is currently marked missing."
        },
        {
          "id": "choose_tab",
          "description": "Select the Details tab to view the requested additional information."
        },
        {
          "id": "sort_recent",
          "description": "Change the result ordering to most recent first."
        },
        {
          "id": "expand_section",
          "description": "Expand the collapsed section containing the requested information."
        },
        {
          "id": "download_text",
          "description": "Download the plain text copy using the visible download link."
        },
        {
          "id": "copy_code",
          "description": "Copy the displayed reference code using the adjacent copy button."
        },
        {
          "id": "wait_loading",
          "description": "Wait for the current loading indicator to finish before taking another action."
        },
        {
          "id": "stop_complete",
          "description": "Stop because the requested task is already complete and confirmed."
        },
        {
          "id": "open_result",
          "description": "Open the first relevant search result in the current tab."
        },
        {
          "id": "enter_query",
          "description": "Enter the requested query in the empty search field."
        },
        {
          "id": "submit_search",
          "description": "Submit the completed search field using the Search button."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice16_02",
      "group_id": "choice16",
      "state": "{\"goal\": \"find the lunar archive schedule\", \"history\": \"The requested query has been typed, but no results are displayed yet.\", \"loading\": false, \"page\": \"document search\", \"search_field\": \"lunar archive schedule\", \"visible_controls\": [\"search field\", \"Search button\"]}",
      "instructions": "Choose the next browser action. The requested query is already in the search field and the Search button is enabled.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "save_draft",
          "description": "Save the completed form as a draft without submitting it."
        },
        {
          "id": "submit_form",
          "description": "Submit the completed form once all required fields are valid."
        },
        {
          "id": "fix_required",
          "description": "Fill the required field that is currently marked missing."
        },
        {
          "id": "choose_tab",
          "description": "Select the Details tab to view the requested additional information."
        },
        {
          "id": "sort_recent",
          "description": "Change the result ordering to most recent first."
        },
        {
          "id": "expand_section",
          "description": "Expand the collapsed section containing the requested information."
        },
        {
          "id": "download_text",
          "description": "Download the plain text copy using the visible download link."
        },
        {
          "id": "copy_code",
          "description": "Copy the displayed reference code using the adjacent copy button."
        },
        {
          "id": "wait_loading",
          "description": "Wait for the current loading indicator to finish before taking another action."
        },
        {
          "id": "stop_complete",
          "description": "Stop because the requested task is already complete and confirmed."
        },
        {
          "id": "open_result",
          "description": "Open the first relevant search result in the current tab."
        },
        {
          "id": "enter_query",
          "description": "Enter the requested query in the empty search field."
        },
        {
          "id": "submit_search",
          "description": "Submit the completed search field using the Search button."
        },
        {
          "id": "go_back",
          "description": "Return to the previous page without changing any data."
        },
        {
          "id": "next_page",
          "description": "Open the next page of the paginated results."
        },
        {
          "id": "close_banner",
          "description": "Dismiss the visible informational banner with its close button."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice16_03",
      "group_id": "choice16",
      "state": "{\"加载中\": false, \"历史\": \"列表已加载，当前没有选中的资料，也没有未保存的修改。\", \"可见控件\": [\"排序菜单\", \"下一页\", \"搜索框\"], \"排序\": \"标题字母顺序\", \"目标\": \"先查看最近更新的资料\", \"页面\": \"资料列表\"}",
      "instructions": "根据共享状态选择下一步浏览器操作。用户只要求调整结果顺序，让最近更新的资料排在最前面。",
      "primitive": "choice",
      "criteria": [
        {
          "id": "choose_tab",
          "description": "Select the Details tab to view the requested additional information."
        },
        {
          "id": "sort_recent",
          "description": "Change the result ordering to most recent first."
        },
        {
          "id": "expand_section",
          "description": "Expand the collapsed section containing the requested information."
        },
        {
          "id": "download_text",
          "description": "Download the plain text copy using the visible download link."
        },
        {
          "id": "copy_code",
          "description": "Copy the displayed reference code using the adjacent copy button."
        },
        {
          "id": "wait_loading",
          "description": "Wait for the current loading indicator to finish before taking another action."
        },
        {
          "id": "stop_complete",
          "description": "Stop because the requested task is already complete and confirmed."
        },
        {
          "id": "open_result",
          "description": "Open the first relevant search result in the current tab."
        },
        {
          "id": "enter_query",
          "description": "Enter the requested query in the empty search field."
        },
        {
          "id": "submit_search",
          "description": "Submit the completed search field using the Search button."
        },
        {
          "id": "go_back",
          "description": "Return to the previous page without changing any data."
        },
        {
          "id": "next_page",
          "description": "Open the next page of the paginated results."
        },
        {
          "id": "close_banner",
          "description": "Dismiss the visible informational banner with its close button."
        },
        {
          "id": "save_draft",
          "description": "Save the completed form as a draft without submitting it."
        },
        {
          "id": "submit_form",
          "description": "Submit the completed form once all required fields are valid."
        },
        {
          "id": "fix_required",
          "description": "Fill the required field that is currently marked missing."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice16_04",
      "group_id": "choice16",
      "state": "{\"保存状态\": \"未保存\", \"名称\": \"灯塔档案室\", \"必填日期\": \"缺失\", \"提示\": \"日期字段显示必填错误，提交按钮暂时不可用。\", \"目标\": \"完成所有必填字段后提交\", \"页面\": \"登记表单\"}",
      "instructions": "选择下一步操作。应先解决表单明确指出的必填字段问题，再尝试提交；不要猜测字段内容。",
      "primitive": "choice",
      "criteria": [
        {
          "id": "download_text",
          "description": "Download the plain text copy using the visible download link."
        },
        {
          "id": "copy_code",
          "description": "Copy the displayed reference code using the adjacent copy button."
        },
        {
          "id": "wait_loading",
          "description": "Wait for the current loading indicator to finish before taking another action."
        },
        {
          "id": "stop_complete",
          "description": "Stop because the requested task is already complete and confirmed."
        },
        {
          "id": "open_result",
          "description": "Open the first relevant search result in the current tab."
        },
        {
          "id": "enter_query",
          "description": "Enter the requested query in the empty search field."
        },
        {
          "id": "submit_search",
          "description": "Submit the completed search field using the Search button."
        },
        {
          "id": "go_back",
          "description": "Return to the previous page without changing any data."
        },
        {
          "id": "next_page",
          "description": "Open the next page of the paginated results."
        },
        {
          "id": "close_banner",
          "description": "Dismiss the visible informational banner with its close button."
        },
        {
          "id": "save_draft",
          "description": "Save the completed form as a draft without submitting it."
        },
        {
          "id": "submit_form",
          "description": "Submit the completed form once all required fields are valid."
        },
        {
          "id": "fix_required",
          "description": "Fill the required field that is currently marked missing."
        },
        {
          "id": "choose_tab",
          "description": "Select the Details tab to view the requested additional information."
        },
        {
          "id": "sort_recent",
          "description": "Change the result ordering to most recent first."
        },
        {
          "id": "expand_section",
          "description": "Expand the collapsed section containing the requested information."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice16_05",
      "group_id": "choice16",
      "state": "{\"active_tab\": \"Overview\", \"goal\": \"read the equipment details\", \"history\": \"The report is open and the Details tab is visible but has not been selected.\", \"loading\": false, \"page\": \"report viewer\", \"tabs\": [\"Overview\", \"Details\", \"History\"]}",
      "instructions": "Choose the browser action that displays the requested equipment details with the fewest additional navigation steps.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stop_complete",
          "description": "Stop because the requested task is already complete and confirmed."
        },
        {
          "id": "open_result",
          "description": "Open the first relevant search result in the current tab."
        },
        {
          "id": "enter_query",
          "description": "Enter the requested query in the empty search field."
        },
        {
          "id": "submit_search",
          "description": "Submit the completed search field using the Search button."
        },
        {
          "id": "go_back",
          "description": "Return to the previous page without changing any data."
        },
        {
          "id": "next_page",
          "description": "Open the next page of the paginated results."
        },
        {
          "id": "close_banner",
          "description": "Dismiss the visible informational banner with its close button."
        },
        {
          "id": "save_draft",
          "description": "Save the completed form as a draft without submitting it."
        },
        {
          "id": "submit_form",
          "description": "Submit the completed form once all required fields are valid."
        },
        {
          "id": "fix_required",
          "description": "Fill the required field that is currently marked missing."
        },
        {
          "id": "choose_tab",
          "description": "Select the Details tab to view the requested additional information."
        },
        {
          "id": "sort_recent",
          "description": "Change the result ordering to most recent first."
        },
        {
          "id": "expand_section",
          "description": "Expand the collapsed section containing the requested information."
        },
        {
          "id": "download_text",
          "description": "Download the plain text copy using the visible download link."
        },
        {
          "id": "copy_code",
          "description": "Copy the displayed reference code using the adjacent copy button."
        },
        {
          "id": "wait_loading",
          "description": "Wait for the current loading indicator to finish before taking another action."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice16_06",
      "group_id": "choice16",
      "state": "{\"提示\": \"申请已经接收，无需重复提交；页面没有加载动画，也没有待处理的错误。\", \"状态\": \"提交成功\", \"目标\": \"提交仓库访问申请\", \"确认编号\": \"CASE-482\", \"页面\": \"申请结果\"}",
      "instructions": "用户要求的申请已经成功提交，并且页面显示确认编号。请选择适当的下一步浏览器操作。",
      "primitive": "choice",
      "criteria": [
        {
          "id": "submit_search",
          "description": "Submit the completed search field using the Search button."
        },
        {
          "id": "go_back",
          "description": "Return to the previous page without changing any data."
        },
        {
          "id": "next_page",
          "description": "Open the next page of the paginated results."
        },
        {
          "id": "close_banner",
          "description": "Dismiss the visible informational banner with its close button."
        },
        {
          "id": "save_draft",
          "description": "Save the completed form as a draft without submitting it."
        },
        {
          "id": "submit_form",
          "description": "Submit the completed form once all required fields are valid."
        },
        {
          "id": "fix_required",
          "description": "Fill the required field that is currently marked missing."
        },
        {
          "id": "choose_tab",
          "description": "Select the Details tab to view the requested additional information."
        },
        {
          "id": "sort_recent",
          "description": "Change the result ordering to most recent first."
        },
        {
          "id": "expand_section",
          "description": "Expand the collapsed section containing the requested information."
        },
        {
          "id": "download_text",
          "description": "Download the plain text copy using the visible download link."
        },
        {
          "id": "copy_code",
          "description": "Copy the displayed reference code using the adjacent copy button."
        },
        {
          "id": "wait_loading",
          "description": "Wait for the current loading indicator to finish before taking another action."
        },
        {
          "id": "stop_complete",
          "description": "Stop because the requested task is already complete and confirmed."
        },
        {
          "id": "open_result",
          "description": "Open the first relevant search result in the current tab."
        },
        {
          "id": "enter_query",
          "description": "Enter the requested query in the empty search field."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice16_07",
      "group_id": "choice16",
      "state": "{\"controls\": [\"Next\", \"Sort\", \"Search\"], \"current_page\": 1, \"goal\": \"inspect the next group of entries\", \"history\": \"The current entries were reviewed. The next-page control is enabled.\", \"loading\": false, \"page\": \"results\", \"page_count\": 4}",
      "instructions": "Choose the next browser action that continues reviewing entries on the following results page. Review note 1: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 2: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 3: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 4: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 5: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 6: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 7: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 8: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 9: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 10: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 11: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 12: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 13: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 14: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 15: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 16: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 17: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 18: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 19: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 20: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 21: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 22: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 23: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 24: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 25: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 26: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 27: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 28: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 29: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 30: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 31: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 32: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 33: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 34: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 35: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 36: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 37: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 38: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 39: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 40: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "close_banner",
          "description": "Dismiss the visible informational banner with its close button."
        },
        {
          "id": "save_draft",
          "description": "Save the completed form as a draft without submitting it."
        },
        {
          "id": "submit_form",
          "description": "Submit the completed form once all required fields are valid."
        },
        {
          "id": "fix_required",
          "description": "Fill the required field that is currently marked missing."
        },
        {
          "id": "choose_tab",
          "description": "Select the Details tab to view the requested additional information."
        },
        {
          "id": "sort_recent",
          "description": "Change the result ordering to most recent first."
        },
        {
          "id": "expand_section",
          "description": "Expand the collapsed section containing the requested information."
        },
        {
          "id": "download_text",
          "description": "Download the plain text copy using the visible download link."
        },
        {
          "id": "copy_code",
          "description": "Copy the displayed reference code using the adjacent copy button."
        },
        {
          "id": "wait_loading",
          "description": "Wait for the current loading indicator to finish before taking another action."
        },
        {
          "id": "stop_complete",
          "description": "Stop because the requested task is already complete and confirmed."
        },
        {
          "id": "open_result",
          "description": "Open the first relevant search result in the current tab."
        },
        {
          "id": "enter_query",
          "description": "Enter the requested query in the empty search field."
        },
        {
          "id": "submit_search",
          "description": "Submit the completed search field using the Search button."
        },
        {
          "id": "go_back",
          "description": "Return to the previous page without changing any data."
        },
        {
          "id": "next_page",
          "description": "Open the next page of the paginated results."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": true
  },
  {
    "request": {
      "id": "choice16_08",
      "group_id": "choice16",
      "state": "{\"必填字段\": \"全部填写\", \"按钮\": [\"保存草稿\", \"正式提交\", \"返回\"], \"状态\": \"存在未保存的修改，当前没有网络请求进行中。\", \"目标\": \"保留当前内容供以后检查，不要正式提交\", \"页面\": \"记录编辑器\"}",
      "instructions": "选择能满足目标的下一步操作。必须保留当前输入，并把记录留在草稿状态。 Review note 1: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 2: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 3: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 4: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 5: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 6: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 7: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 8: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 9: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 10: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 11: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 12: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 13: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 14: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 15: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 16: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 17: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 18: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 19: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 20: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 21: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 22: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 23: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 24: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 25: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 26: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 27: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 28: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 29: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 30: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 31: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 32: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 33: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 34: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 35: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 36: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 37: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 38: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 39: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action. Review note 40: the visible state is authoritative; preserve the stated goal, avoid repeating completed steps, and choose only the next permitted action.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "fix_required",
          "description": "Fill the required field that is currently marked missing."
        },
        {
          "id": "choose_tab",
          "description": "Select the Details tab to view the requested additional information."
        },
        {
          "id": "sort_recent",
          "description": "Change the result ordering to most recent first."
        },
        {
          "id": "expand_section",
          "description": "Expand the collapsed section containing the requested information."
        },
        {
          "id": "download_text",
          "description": "Download the plain text copy using the visible download link."
        },
        {
          "id": "copy_code",
          "description": "Copy the displayed reference code using the adjacent copy button."
        },
        {
          "id": "wait_loading",
          "description": "Wait for the current loading indicator to finish before taking another action."
        },
        {
          "id": "stop_complete",
          "description": "Stop because the requested task is already complete and confirmed."
        },
        {
          "id": "open_result",
          "description": "Open the first relevant search result in the current tab."
        },
        {
          "id": "enter_query",
          "description": "Enter the requested query in the empty search field."
        },
        {
          "id": "submit_search",
          "description": "Submit the completed search field using the Search button."
        },
        {
          "id": "go_back",
          "description": "Return to the previous page without changing any data."
        },
        {
          "id": "next_page",
          "description": "Open the next page of the paginated results."
        },
        {
          "id": "close_banner",
          "description": "Dismiss the visible informational banner with its close button."
        },
        {
          "id": "save_draft",
          "description": "Save the completed form as a draft without submitting it."
        },
        {
          "id": "submit_form",
          "description": "Submit the completed form once all required fields are valid."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": true
  },
  {
    "request": {
      "id": "choice_small_01",
      "group_id": "choice_small",
      "state": "Review status: complete\nRequired checks: all passed\nRequested outcome: preserve the approved draft for tomorrow\nCurrent record: editable and unsaved\nAvailable actions: save a draft or submit it for final processing.",
      "instructions": "Choose the action that preserves the work without starting final processing.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "save",
          "description": "Save the current content as a draft."
        },
        {
          "id": "submit",
          "description": "Submit the current content for final processing."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_small_02",
      "group_id": "choice_small",
      "state": "A local workflow has two steps. The first step extracts text from an uploaded synthetic note. The second checks that text for missing fields. Extraction has finished successfully, and the check has not started.",
      "instructions": "Choose the next step required to finish this workflow.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "extract",
          "description": "Repeat text extraction from the same note."
        },
        {
          "id": "check",
          "description": "Check the extracted text for missing fields."
        },
        {
          "id": "finish",
          "description": "Mark the workflow finished before checking it."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_small_03",
      "group_id": "choice_small",
      "state": "任务状态：等待审核\n必需附件：已上传\n审核结果：尚未产生\n用户目标：只有审核通过后才能归档\n当前记录没有丢失，也没有需要重新上传的附件。",
      "instructions": "根据当前状态选择合适的下一步，不要跳过用户要求的审核。",
      "primitive": "choice",
      "criteria": [
        {
          "id": "review",
          "description": "Start the required review of the uploaded attachment."
        },
        {
          "id": "archive",
          "description": "Archive the item immediately without review."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_small_04",
      "group_id": "choice_small",
      "state": "A short editing task contains a paragraph and a table. The paragraph has been checked. The table still contains an empty required cell, and the completion rule requires every cell in that column to be filled.",
      "instructions": "Choose the action that addresses the remaining blocker before completion.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "finish",
          "description": "Declare the editing task complete."
        },
        {
          "id": "repeat",
          "description": "Review the already checked paragraph again."
        },
        {
          "id": "fill",
          "description": "Fill the empty required table cell using the available source."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_small_05",
      "group_id": "choice_small",
      "state": "下载状态：正在进行\n进度：百分之六十五\n错误信息：无\n目标：取得完整文件后再检查内容\n页面仍然显示活动的进度条，当前没有完成通知。",
      "instructions": "请选择现在应该采取的动作，避免把未完成的文件当成完整结果。",
      "primitive": "choice",
      "criteria": [
        {
          "id": "wait",
          "description": "Wait for the current download to finish."
        },
        {
          "id": "inspect",
          "description": "Inspect the file as though the download were complete."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_small_06",
      "group_id": "choice_small",
      "state": "The fictional inventory list contains five entries. A filter currently hides archived entries. The user asks to inspect an entry known to be archived. The list itself has loaded successfully with no active request.",
      "instructions": "Choose the action that makes the requested archived entry visible.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "refresh",
          "description": "Reload the unchanged filtered list."
        },
        {
          "id": "filter",
          "description": "Change the filter to include archived entries."
        },
        {
          "id": "remove",
          "description": "Delete the visible active entries."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_small_07",
      "group_id": "choice_small",
      "state": "表单状态：已填写\n校验状态：通过\n用户指令：仅预览，不发送\n当前页面：编辑页\n可见按钮：预览、发送。没有后台提交任务，也没有其他待处理的页面。",
      "instructions": "用户只希望检查最终外观。请选择满足这一限制的操作。",
      "primitive": "choice",
      "criteria": [
        {
          "id": "preview",
          "description": "Open the preview without sending the form."
        },
        {
          "id": "send",
          "description": "Send the form to its final destination."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_small_08",
      "group_id": "choice_small",
      "state": "A workflow routes a synthetic request to one of three queues. The record type is maintenance, its urgency is routine, and the routing rules say routine maintenance belongs in the maintenance queue rather than the emergency or research queues.",
      "instructions": "Choose the queue specified by the routing rules for this record.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "research",
          "description": "Place the record in the research queue."
        },
        {
          "id": "maintenance",
          "description": "Place the record in the routine maintenance queue."
        },
        {
          "id": "emergency",
          "description": "Place the record in the emergency queue."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_01",
      "group_id": "choice_mid",
      "state": "{\"constraint\": \"execute exactly one unfinished stage before reporting completion\", \"inputs\": \"all required material is ready\", \"only_pending_stage\": \"stage_2\", \"other_stages\": \"all completed and must not be repeated\", \"stages\": 4, \"workflow\": \"document processing 1\"}",
      "instructions": "Choose the only workflow stage that still needs execution. Treat the explicit pending-stage field as authoritative.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_02",
      "group_id": "choice_mid",
      "state": "{\"其他阶段\": \"均已完成，无需重复执行\", \"唯一待处理阶段\": \"stage_4\", \"工作流\": \"文档处理2\", \"约束\": \"本次只能执行一个阶段，不能提前宣布完成\", \"输入\": \"所有必要资料已经准备好\", \"阶段总数\": 5}",
      "instructions": "选择唯一尚未完成的工作流阶段。不要重复已完成阶段，也不要更改共享状态中给出的执行顺序。",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_5",
          "description": "Execute workflow stage 5, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_03",
      "group_id": "choice_mid",
      "state": "{\"constraint\": \"execute exactly one unfinished stage before reporting completion\", \"inputs\": \"all required material is ready\", \"only_pending_stage\": \"stage_6\", \"other_stages\": \"all completed and must not be repeated\", \"stages\": 6, \"workflow\": \"document processing 3\"}",
      "instructions": "Choose the only workflow stage that still needs execution. Treat the explicit pending-stage field as authoritative.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_5",
          "description": "Execute workflow stage 5, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_6",
          "description": "Execute workflow stage 6, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_04",
      "group_id": "choice_mid",
      "state": "{\"constraint\": \"execute exactly one unfinished stage before reporting completion\", \"inputs\": \"all required material is ready\", \"only_pending_stage\": \"stage_1\", \"other_stages\": \"all completed and must not be repeated\", \"stages\": 7, \"workflow\": \"document processing 4\"}",
      "instructions": "Choose the only workflow stage that still needs execution. Treat the explicit pending-stage field as authoritative.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_5",
          "description": "Execute workflow stage 5, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_6",
          "description": "Execute workflow stage 6, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_7",
          "description": "Execute workflow stage 7, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_05",
      "group_id": "choice_mid",
      "state": "{\"constraint\": \"execute exactly one unfinished stage before reporting completion\", \"inputs\": \"all required material is ready\", \"only_pending_stage\": \"stage_2\", \"other_stages\": \"all completed and must not be repeated\", \"stages\": 8, \"workflow\": \"document processing 5\"}",
      "instructions": "Choose the only workflow stage that still needs execution. Treat the explicit pending-stage field as authoritative.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_5",
          "description": "Execute workflow stage 5, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_6",
          "description": "Execute workflow stage 6, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_7",
          "description": "Execute workflow stage 7, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_8",
          "description": "Execute workflow stage 8, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_06",
      "group_id": "choice_mid",
      "state": "{\"其他阶段\": \"均已完成，无需重复执行\", \"唯一待处理阶段\": \"stage_4\", \"工作流\": \"文档处理6\", \"约束\": \"本次只能执行一个阶段，不能提前宣布完成\", \"输入\": \"所有必要资料已经准备好\", \"阶段总数\": 4}",
      "instructions": "选择唯一尚未完成的工作流阶段。不要重复已完成阶段，也不要更改共享状态中给出的执行顺序。",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_07",
      "group_id": "choice_mid",
      "state": "{\"constraint\": \"execute exactly one unfinished stage before reporting completion\", \"inputs\": \"all required material is ready\", \"only_pending_stage\": \"stage_4\", \"other_stages\": \"all completed and must not be repeated\", \"stages\": 5, \"workflow\": \"document processing 7\"}",
      "instructions": "Choose the only workflow stage that still needs execution. Treat the explicit pending-stage field as authoritative.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_5",
          "description": "Execute workflow stage 5, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_08",
      "group_id": "choice_mid",
      "state": "{\"constraint\": \"execute exactly one unfinished stage before reporting completion\", \"inputs\": \"all required material is ready\", \"only_pending_stage\": \"stage_4\", \"other_stages\": \"all completed and must not be repeated\", \"stages\": 6, \"workflow\": \"document processing 8\"}",
      "instructions": "Choose the only workflow stage that still needs execution. Treat the explicit pending-stage field as authoritative.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_5",
          "description": "Execute workflow stage 5, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_6",
          "description": "Execute workflow stage 6, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_09",
      "group_id": "choice_mid",
      "state": "{\"constraint\": \"execute exactly one unfinished stage before reporting completion\", \"inputs\": \"all required material is ready\", \"only_pending_stage\": \"stage_4\", \"other_stages\": \"all completed and must not be repeated\", \"stages\": 7, \"workflow\": \"document processing 9\"}",
      "instructions": "Choose the only workflow stage that still needs execution. Treat the explicit pending-stage field as authoritative.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_5",
          "description": "Execute workflow stage 5, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_6",
          "description": "Execute workflow stage 6, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_7",
          "description": "Execute workflow stage 7, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_10",
      "group_id": "choice_mid",
      "state": "{\"其他阶段\": \"均已完成，无需重复执行\", \"唯一待处理阶段\": \"stage_4\", \"工作流\": \"文档处理10\", \"约束\": \"本次只能执行一个阶段，不能提前宣布完成\", \"输入\": \"所有必要资料已经准备好\", \"阶段总数\": 8}",
      "instructions": "选择唯一尚未完成的工作流阶段。不要重复已完成阶段，也不要更改共享状态中给出的执行顺序。",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_5",
          "description": "Execute workflow stage 5, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_6",
          "description": "Execute workflow stage 6, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_7",
          "description": "Execute workflow stage 7, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_8",
          "description": "Execute workflow stage 8, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_11",
      "group_id": "choice_mid",
      "state": "{\"constraint\": \"execute exactly one unfinished stage before reporting completion\", \"inputs\": \"all required material is ready\", \"only_pending_stage\": \"stage_2\", \"other_stages\": \"all completed and must not be repeated\", \"stages\": 4, \"workflow\": \"document processing 11\"}",
      "instructions": "Choose the only workflow stage that still needs execution. Treat the explicit pending-stage field as authoritative.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "choice_mid_12",
      "group_id": "choice_mid",
      "state": "{\"constraint\": \"execute exactly one unfinished stage before reporting completion\", \"inputs\": \"all required material is ready\", \"only_pending_stage\": \"stage_4\", \"other_stages\": \"all completed and must not be repeated\", \"stages\": 5, \"workflow\": \"document processing 12\"}",
      "instructions": "Choose the only workflow stage that still needs execution. Treat the explicit pending-stage field as authoritative.",
      "primitive": "choice",
      "criteria": [
        {
          "id": "stage_1",
          "description": "Execute workflow stage 1, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_2",
          "description": "Execute workflow stage 2, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_3",
          "description": "Execute workflow stage 3, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_4",
          "description": "Execute workflow stage 4, using the prepared input and preserving completed work."
        },
        {
          "id": "stage_5",
          "description": "Execute workflow stage 5, using the prepared input and preserving completed work."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_01",
      "group_id": "noul",
      "state": "Record status: approved\nRequired attachments: three\nUploaded attachments: three\nValidation errors: zero\nArchive rule: approval and every attachment are required\nThe listed values are current and complete.",
      "instructions": "The record meets the stated requirements for archiving.",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_02",
      "group_id": "noul",
      "state": "Record status: awaiting review\nRequired attachments: three\nUploaded attachments: three\nValidation errors: zero\nArchive rule: approval and every attachment are required\nNo reviewer has approved the record yet.",
      "instructions": "The record meets the stated requirements for archiving.",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_03",
      "group_id": "noul",
      "state": "任务清单共有四项。前三项已经完成，第四项仍然等待执行。完成规则要求四项全部完成后才能关闭任务。当前没有后台执行，也没有遗漏的结果通知。",
      "instructions": "命题：根据所述完成规则，现在可以关闭整个任务。",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "zh",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_04",
      "group_id": "noul",
      "state": "任务清单共有四项。四项都已经执行并通过检查，系统没有待处理项。完成规则要求四项全部完成后才能关闭任务，当前结果已经满足这一要求。",
      "instructions": "命题：根据所述完成规则，现在可以关闭整个任务。",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "zh",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_05",
      "group_id": "noul",
      "state": "The generated fictional report is eight pages long. Its page counter displays page eight of eight. The next-page control is disabled, the document is fully loaded, and the user wants to know whether another report page remains.",
      "instructions": "At least one additional page of this report remains after the current page.",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_06",
      "group_id": "noul",
      "state": "The generated fictional report is eight pages long. Its page counter displays page three of eight. The next-page control is enabled, the document is fully loaded, and the user wants to know whether another report page remains.",
      "instructions": "At least one additional page of this report remains after the current page.",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_07",
      "group_id": "noul",
      "state": "库存记录：合成部件\n可用数量：十二\n本次需求：九\n保留数量：零\n规则：可用数量不少于需求数量即可满足本次需求。所有数量采用相同单位，记录刚刚更新。",
      "instructions": "命题：当前可用库存足够满足本次需求。",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_08",
      "group_id": "noul",
      "state": "库存记录：合成部件\n可用数量：六\n本次需求：九\n保留数量：零\n规则：可用数量不少于需求数量即可满足本次需求。没有正在到货的额外库存。",
      "instructions": "命题：当前可用库存足够满足本次需求。",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "zh",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_09",
      "group_id": "noul",
      "state": "A fictional processing job has a final result file. The verifier compared every required field with the input record and found no discrepancies. The success rule requires an output file and a successful verification, and no further checks are listed.",
      "instructions": "The processing job satisfies its stated success rule.",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_10",
      "group_id": "noul",
      "state": "A fictional processing job has a final result file. The verifier has not yet examined that file. The success rule requires both an output file and a successful verification, and the current status explicitly says verification pending.",
      "instructions": "The processing job satisfies its stated success rule.",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_11",
      "group_id": "noul",
      "state": "Filter state: archived entries included\nTarget entry state: archived\nSearch text: empty\nOther restrictions: none\nThe results have finished loading, and the target entry belongs to the current collection.",
      "instructions": "The described filters permit the target archived entry to appear.",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "noul_12",
      "group_id": "noul",
      "state": "Filter state: archived entries excluded\nTarget entry state: archived\nSearch text: empty\nOther restrictions: none\nThe results have finished loading, and the target entry belongs to the current collection.",
      "instructions": "The described filters permit the target archived entry to appear.",
      "primitive": "noul",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": true,
    "zoo_only": false
  },
  {
    "request": {
      "id": "score_level_01",
      "group_id": "score_level",
      "state": "A fictional review rubric defines readiness as passing at least three of four checks. The item passed checks one, two, and four; it failed check three. All four results are final, and no additional criteria apply.",
      "instructions": "The item reaches the rubric's readiness level.",
      "primitive": "score_level",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "score_level_02",
      "group_id": "score_level",
      "state": "A fictional review rubric defines readiness as passing at least three of four checks. The item passed checks one and four; it failed checks two and three. All four results are final, and no additional criteria apply.",
      "instructions": "The item reaches the rubric's readiness level.",
      "primitive": "score_level",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "score_level_03",
      "group_id": "score_level",
      "state": "评分规则规定：完整性达到四项中的四项才属于完整级别。当前记录的名称、日期、类别和说明四项都已填写，格式检查通过，没有其他必需字段。",
      "instructions": "命题：当前记录达到规则定义的完整级别。",
      "primitive": "score_level",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "zh",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "score_level_04",
      "group_id": "score_level",
      "state": "评分规则规定：完整性达到四项中的四项才属于完整级别。当前记录填写了名称、日期和类别，但说明仍然为空，其他已填字段的格式检查通过。",
      "instructions": "命题：当前记录达到规则定义的完整级别。",
      "primitive": "score_level",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "zh",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "score_level_05",
      "group_id": "score_level",
      "state": "A fictional workflow rubric assigns the verified level only when both data extraction and an independent check have finished. Extraction succeeded, the independent check succeeded, and the latest status contains no unresolved issue.",
      "instructions": "The workflow reaches the rubric's verified level.",
      "primitive": "score_level",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "score_level_06",
      "group_id": "score_level",
      "state": "A fictional workflow rubric assigns the verified level only when both data extraction and an independent check have finished. Extraction succeeded, but the independent check is still queued and has produced no result.",
      "instructions": "The workflow reaches the rubric's verified level.",
      "primitive": "score_level",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "score_level_07",
      "group_id": "score_level",
      "state": "A synthetic submission has a quality threshold of ninety points out of one hundred. Its final score is ninety-two, all scoring fields are present, and the rubric states that meeting or exceeding the threshold qualifies for the upper level.",
      "instructions": "The submission reaches the rubric's upper quality level.",
      "primitive": "score_level",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  },
  {
    "request": {
      "id": "score_level_08",
      "group_id": "score_level",
      "state": "A synthetic submission has a quality threshold of ninety points out of one hundred. Its final score is eighty-six, all scoring fields are present, and the rubric states that meeting or exceeding the threshold qualifies for the upper level.",
      "instructions": "The submission reaches the rubric's upper quality level.",
      "primitive": "score_level",
      "criteria": [
        {
          "id": "yes",
          "description": "The stated proposition is true."
        },
        {
          "id": "no",
          "description": "The stated proposition is false."
        }
      ]
    },
    "language": "en",
    "structured_state": false,
    "zoo_only": false
  }
]''')


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    from huggingface_hub import snapshot_download
    snapshot = args.snapshot
    if snapshot is None:
        patterns = (["*.json", "*.jinja", "openjet_runtime/*", "evaluation/runtime-smoke.json"]
                    if args.replay_fixtures else None)
        snapshot = Path(snapshot_download(args.hf_id, revision=args.revision,
                                         max_workers=8, allow_patterns=patterns))
    snapshot = snapshot.resolve()
    if snapshot.name != args.revision:
        raise ValueError("snapshot directory must name the pinned full commit")
    smoke_path = snapshot / "evaluation/runtime-smoke.json"
    expected = json.loads(smoke_path.read_text())["runtime_sha256"]
    package = WORK / "author/openjet_runtime"
    package.mkdir(parents=True, exist_ok=True)
    files = []
    for source in sorted((snapshot / "openjet_runtime").glob("*.py")):
        target = package / source.name
        shutil.copyfile(source, target)
        matches = [key for key in expected if Path(key).name == source.name]
        assert len(matches) == 1
        assert sha256(target) == sha256(source) == expected[matches[0]]
        files.append({"path": str(target), "source": str(source), "sha256": sha256(target),
                      "runtime_smoke_key": matches[0], "unmodified": True})
    assert len(files) == len(expected) == 5
    atomic_json(WORK / "author_sources.json", {"status": "PASS", "hf_id": args.hf_id,
                "revision": args.revision, "runtime_smoke_sha256": sha256(smoke_path), "files": files})
    sys.path.insert(0, str(WORK / "author"))
    from openjet_runtime import OpenJet
    from openjet_runtime.contracts import BINARY_CRITERIA, LABELS, PROMPT_VERSION
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    compiler = OpenJet.__new__(OpenJet)
    compiler.tokenizer = tokenizer
    compiler.model = SimpleNamespace(config=AutoConfig.from_pretrained(snapshot, local_files_only=True))
    compiler.max_length = 8192
    rows = copy.deepcopy(REQUESTS)
    for row in rows:
        req = row["request"]
        ids, label_ids = compiler.compile(req)
        state_tokens = len(tokenizer.encode(req["state"], add_special_tokens=False))
        assert 20 <= state_tokens <= 600
        assert (1500 <= len(ids) <= 2000) if row["zoo_only"] else len(ids) <= 1024
        assert req["primitive"] == "choice" or req["criteria"] == BINARY_CRITERIA
        row.update(ids=ids, label_ids=label_ids, state_tokens=state_tokens, tokens=len(ids))
    counts = {"requests": len(rows), "rows": len(rows),
              **{kind: sum(r["request"]["primitive"] == kind for r in rows)
                 for kind in ("choice", "noul", "score_level")},
              "choice_16": sum(r["request"]["primitive"] == "choice" and len(r["request"]["criteria"]) == 16 for r in rows),
              "choice_2_3": sum(r["request"]["primitive"] == "choice" and len(r["request"]["criteria"]) in (2, 3) for r in rows),
              "choice_4_8": sum(r["request"]["primitive"] == "choice" and 4 <= len(r["request"]["criteria"]) <= 8 for r in rows),
              "chinese": sum(bool(re.search(r"[\u3400-\u9fff]", r["request"]["state"] + r["request"]["instructions"])) for r in rows),
              "structured_states": sum(r["structured_state"] for r in rows),
              "zoo_only": sum(r["zoo_only"] for r in rows),
              "tokens_min": min(r["tokens"] for r in rows), "tokens_max": max(r["tokens"] for r in rows),
              "state_tokens_min": min(r["state_tokens"] for r in rows),
              "state_tokens_max": max(r["state_tokens"] for r in rows), "boundary_assertions_all": True}
    assert len(rows) == 48 and counts["choice"] >= 26 and counts["noul"] >= 12 and counts["score_level"] >= 8
    assert min(counts[k] for k in ("choice_16", "choice_2_3", "choice_4_8")) >= 6
    assert counts["chinese"] >= 8 and counts["structured_states"] >= 6 and counts["zoo_only"] == 2
    inputs = {"schema": "coreai-letter-inputs/1", "source": {"hf_id": args.hf_id, "revision": args.revision},
              "snapshot": str(snapshot), "prompt_version": PROMPT_VERSION, "letters": list(LABELS),
              "temperature": 1.0, "chat_template_sha256": sha256(snapshot / "chat_template.jinja"),
              "rows": rows, "summary": counts}
    atomic_json(WORK / "oracle_inputs.json", inputs)
    return inputs, compiler


def replay(args, inputs, compiler):
    import torch
    from openjet_runtime.contracts import render_prompt
    fixture = json.loads(args.replay_fixtures.read_text())
    assert fixture["schema"] == "coreai-letter-fixtures/1" and fixture["result"] == "PASS"
    assert fixture["source"]["hf_id"] == args.hf_id and fixture["source"]["revision"] == args.revision
    assert fixture["source"]["transformers"] == "5.16.1"
    assert fixture["chat_template_sha256"] == inputs["chat_template_sha256"]
    assert fixture["requests"] == [row["request"] for row in inputs["rows"]]
    assert len(fixture["rows"]) == 48
    for row, inp in zip(fixture["rows"], inputs["rows"]):
        assert row["request"] == inp["request"] and row["ids"] == inp["ids"]
        assert row["label_ids"] == inp["label_ids"] and row["slot"] == len(inp["ids"]) - 1
        assert row["state_tokens"] == inp["state_tokens"] and row["tokens"] == len(inp["ids"])
        expected_labels = list("ABCDEFGHIJKLMNOP"[:len(row["request"]["criteria"])])
        assert row["labels"] == expected_labels
        logits = torch.tensor(row["raw_logits"], dtype=torch.float32)
        assert torch.isfinite(logits).all() and logits.softmax(-1).tolist() == row["p_oracle"]
        keyed = dict(zip([c["id"] for c in row["request"]["criteria"]], row["p_oracle"]))
        assert row["api"]["probabilities"] == keyed and row["api"]["calibrated"] is False
        assert row["api"]["executed_layers"] == 32 and row["api"]["projection"] == "full_head"
        assert row["api"]["logits"] == row["raw_logits"] and int(logits.argmax()) == row["argmax"]
        assert row["depth_execution"]["high"] == list(range(32))
        assert row["depth_execution"]["low"] == list(range(16))
    by_id = {row["id"]: row for row in fixture["rows"]}
    for example in fixture["prompt_examples"]:
        row = by_id[example["id"]]
        assert example["rendered_prompt"] == render_prompt(row["request"])
        assert example["decoded_ids"] == compiler.tokenizer.decode(row["ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.replay_fixtures, args.out)
    assert sha256(args.out) == sha256(args.replay_fixtures)
    proof = {"schema": "coreai-letter-oracle-replay/1", "result": "PASS", "rows": 48,
             "source": inputs["source"], "input": str(args.replay_fixtures), "output": str(args.out),
             "fixtures_sha256": sha256(args.out), "byte_identical": True, "compiled_ids_equal": 48,
             "label_ids_equal": 48, "api_probability_equal": 48, "model_inference_rerun": False,
             "script_sha256": sha256(Path(__file__)), "author_sources": str(WORK / "author_sources.json"),
             "summary": inputs["summary"], "generated_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(WORK / "oracle_replay.json", proof)
    print(json.dumps(proof, ensure_ascii=False, indent=2), flush=True)


def run_fresh(args, inputs):
    import torch
    import transformers
    from openjet_runtime import OpenJet
    from openjet_runtime.contracts import LABELS, PROMPT_VERSION, label_mapping, render_prompt, render_prompt_parts
    start = time.monotonic()
    deadline = args.deadline_epoch
    def check_deadline():
        if time.time() >= deadline:
            raise TimeoutError("oracle wall-clock deadline reached")
    snapshot = Path(inputs["snapshot"])
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    record = {
        "schema": "coreai-letter-fixtures/1", "status": "RUNNING", "result": "RUNNING",
        "source": {**inputs["source"], "oracle": "author's vendored unchanged openjet_runtime.OpenJet",
                   "transformers": transformers.__version__, "torch": torch.__version__,
                   "device": "cpu", "dtype": "float32", "vendor_evidence": str(WORK / "author_sources.json")},
        "environment": {"device": "CPU",
                        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                        "python": sys.version, "pid": os.getpid(), "fence": "OPEN",
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("transformers", "torch", "safetensors", "huggingface_hub", "numpy", "tokenizers")}},
        "letters": list(LABELS), "temperature": 1.0, "prompt_version": PROMPT_VERSION,
        "chat_template_sha256": inputs["chat_template_sha256"],
        "requests": [row["request"] for row in inputs["rows"]],
        "rows": [], "prompt_examples": [], "summary": {**inputs["summary"], "completed_rows": 0},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "method": "Unchanged author OpenJet.from_pretrained(snapshot, device='cpu', dtype='float32'); "
                  "compile and high/low decide for every row. Independent fp32 softmax on exposed high logits; "
                  "exact API-probability equality. Non-mutating decoder-layer pre-hooks count execution; "
                  "no head/global hooks and no exporter/runtime modifications.",
    }
    output = args.out
    atomic_json(output, record)
    phase = "load"
    active_row = None
    try:
        print("Loading author's unchanged runtime on CPU float32", flush=True)
        tload = time.monotonic()
        jet = OpenJet.from_pretrained(snapshot, device="cpu", dtype="float32")
        record["load_seconds"] = time.monotonic() - tload
        assert jet.depths == {"low": 16, "high": 32}
        assert str(jet.device) == "cpu"
        assert jet.model.get_input_embeddings().weight.dtype == torch.float32
        assert jet.model.get_output_embeddings().weight.dtype == torch.float32
        assert jet.model.get_input_embeddings().weight.data_ptr() == jet.model.get_output_embeddings().weight.data_ptr()
        assert list(jet.model.get_output_embeddings().weight.shape) == [248320, 2560]
        record["model"] = {"depths": jet.depths, "embedding_dtype": str(jet.model.get_input_embeddings().weight.dtype),
                           "head_dtype": str(jet.model.get_output_embeddings().weight.dtype), "tied_head": True,
                           "vocab_size": int(jet.model.get_output_embeddings().weight.shape[0]),
                           "head_shape": list(jet.model.get_output_embeddings().weight.shape)}
        print(f"Loaded in {record['load_seconds']:.1f} s; head tied, depth high=32 / low=16", flush=True)
        seen = []
        handles = []
        for index, layer in enumerate(jet.wrapper.backbone.layers):
            def hook(_module, _inputs, index=index):
                seen.append(index)
            handles.append(layer.register_forward_pre_hook(hook))
        for idx, inp in enumerate(inputs["rows"]):
            check_deadline()
            row_start = time.monotonic()
            req = inp["request"]
            active_row = req["id"]
            phase = "compile"
            ids, label_ids = jet.compile(req)
            assert ids == inp["ids"] and label_ids == inp["label_ids"]
            labels = list(label_mapping(req))
            criteria_ids = [criterion["id"] for criterion in req["criteria"]]
            phase = "high"
            seen.clear()
            high_start = time.monotonic()
            high = jet.decide(req, effort="high")
            high_seconds = time.monotonic() - high_start
            high_layers = seen.copy()
            assert high_layers == list(range(32)), (active_row, high_layers)
            assert high["executed_layers"] == 32 and high["projection"] == "full_head"
            assert high["prompt_tokens"] == len(ids) and high["calibrated"] is False
            logits = torch.tensor(high["logits"], dtype=torch.float32)
            assert logits.shape == (len(labels),) and bool(torch.isfinite(logits).all())
            probabilities = logits.softmax(-1).tolist()
            keyed = dict(zip(criteria_ids, probabilities))
            assert keyed == high["probabilities"], (active_row, keyed, high["probabilities"])
            argmax = int(logits.argmax())
            if req["primitive"] == "choice":
                assert high["choice"] == criteria_ids[argmax]
            else:
                assert high["yes_probability"] == keyed["yes"]
            check_deadline()
            phase = "low"
            seen.clear()
            low_start = time.monotonic()
            low = jet.decide(req, effort="low")
            low_seconds = time.monotonic() - low_start
            low_layers = seen.copy()
            assert low_layers == list(range(16)), (active_row, low_layers)
            assert low["executed_layers"] == 16 and low["projection"] == "candidate_rows"
            assert low["prompt_tokens"] == len(ids) and low["calibrated"] is False
            low_logits = torch.tensor(low["logits"], dtype=torch.float32)
            assert bool(torch.isfinite(low_logits).all())
            low_probabilities = low_logits.softmax(-1).tolist()
            assert dict(zip(criteria_ids, low_probabilities)) == low["probabilities"]
            sorted_p = sorted(probabilities, reverse=True)
            row = {
                "id": active_row, "request_id": active_row, "request": req,
                "primitive": req["primitive"], "labels": labels, "label_strings": labels,
                "criteria_ids": criteria_ids, "nopts": len(labels), "ids": ids,
                "slot": len(ids) - 1, "label_ids": label_ids, "raw_logits": high["logits"],
                "p_oracle": probabilities, "argmax": argmax, "argmax_label": labels[argmax],
                "argmax_criterion_id": criteria_ids[argmax],
                "top2_margin": sorted_p[0] - sorted_p[1], "tokens": len(ids),
                "state_tokens": inp["state_tokens"], "language": inp["language"],
                "structured_state": inp["structured_state"], "zoo_only": inp["zoo_only"],
                "low_effort": {"logits": low["logits"], "probabilities": low["probabilities"],
                               "p_ordered": low_probabilities, "api": low, "evidence_only": True,
                               "not_the_bundle_path": True},
                "api": high, "api_probability_equal": True, "label_boundary_valid": True,
                "depth_execution": {"high": high_layers, "low": low_layers,
                                    "observed_by": "decoder-layer forward pre-hooks; no mutation"},
                "numeric": {"dtype": "float32", "finite": True,
                            "logits_min": float(logits.min()), "logits_max": float(logits.max()),
                            "logits_abs_max": float(logits.abs().max()),
                            "logits_mean_abs": float(logits.abs().mean()),
                            "probability_sum": sum(probabilities)},
                "wall_seconds": {"high": high_seconds, "low": low_seconds,
                                 "row": time.monotonic() - row_start},
            }
            if idx in (0, 2, 6):
                rendered = render_prompt(req)
                prefix, suffix = render_prompt_parts(req)
                chat = jet.tokenizer.apply_chat_template(
                    [{"role": "user", "content": rendered}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                decoded = jet.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                state_end = chat.index(req["state"]) + len(req["state"])
                example = {"id": active_row, "decoded_ids": decoded, "compiled_chat": chat,
                           "rendered_prompt": rendered, "tail_after_state": "\n\n" + suffix,
                           "compiled_tail_after_state": chat[state_end:],
                           "task_json_and_answer_suffix": suffix,
                           "decoded_equals_compiled_chat": decoded == chat}
                assert prefix == "Shared state:\n" + req["state"] + "\n\n"
                assert chat[state_end:].startswith("\n\n" + suffix)
                row["prompt_example"] = example
                record["prompt_examples"].append(example)
            record["rows"].append(row)
            record["summary"]["completed_rows"] = len(record["rows"])
            record["summary"]["api_probability_equal_count"] = len(record["rows"])
            record["summary"]["boundary_valid_count"] = len(record["rows"])
            record["wall_seconds"] = time.monotonic() - start
            record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            atomic_json(output, record)
            print(f"{idx + 1:02}/48 {active_row} tokens={len(ids)} high={high_seconds:.1f}s low={low_seconds:.1f}s "
                  f"label={labels[argmax]} p={probabilities[argmax]:.9f} margin={row['top2_margin']:.9f}", flush=True)
        for handle in handles:
            handle.remove()
        rows = record["rows"]
        assert len(rows) == len(inputs["rows"]) == 48
        record["status"] = record["result"] = "PASS"
        record["summary"].update(
            finite_all=all(r["numeric"]["finite"] for r in rows),
            raw_logits_min=min(r["numeric"]["logits_min"] for r in rows),
            raw_logits_max=max(r["numeric"]["logits_max"] for r in rows),
            raw_logits_abs_max=max(r["numeric"]["logits_abs_max"] for r in rows),
            raw_logits_mean_of_row_mean_abs=sum(r["numeric"]["logits_mean_abs"] for r in rows) / len(rows),
            min_top2_margin=min(r["top2_margin"] for r in rows),
            near_ties=[{"id": r["id"], "top2_margin": r["top2_margin"]} for r in rows if r["top2_margin"] < 0.02],
            full_depth_observed_all=all(r["depth_execution"]["high"] == list(range(32)) for r in rows),
            low_depth_observed_all=all(r["depth_execution"]["low"] == list(range(16)) for r in rows),
            low_effort_evidence_only=True,
            high_low_same_argmax=sum(r["argmax"] == max(range(r["nopts"]), key=lambda i: r["low_effort"]["p_ordered"][i]) for r in rows),
        )
        print("PASS all 48 high/low rows, exact API probabilities, boundary and executed-layer assertions", flush=True)
    except BaseException as exc:
        record["status"] = record["result"] = "BLOCKED" if isinstance(exc, TimeoutError) else "FAIL"
        record["failure"] = {"phase": phase, "row": active_row, "type": type(exc).__name__,
                             "message": str(exc), "traceback": traceback.format_exc()}
        atomic_json(WORK / "oracle_failure.json", record["failure"])
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        record["wall_seconds"] = time.monotonic() - start
        record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output, record)
        atomic_json(WORK / "oracle_execution.json", {
            k: v for k, v in record.items() if k not in ("rows", "requests", "prompt_examples")})



def main():
    global WORK
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf-id", default=DEFAULT_HF_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--out", type=Path, default=Path("fixtures-apus-openjev-v1-4b.json"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--replay-fixtures", type=Path, help="validate/copy accepted fixture; no model inference")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--deadline-epoch", type=float, default=time.time() + 3 * 3600)
    args = parser.parse_args()
    args.out = args.out.resolve()
    WORK = args.work_dir.resolve() if args.work_dir else args.out.parent / ("." + args.out.stem + "-work")
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import torch
    import transformers
    assert transformers.__version__ == "5.16.1"
    assert torch.__version__.split("+")[0] == "2.9.0"
    def timeout(_sig, _frame):
        raise TimeoutError("oracle wall-clock deadline reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, args.deadline_epoch - time.time()))
    inputs, compiler = prepare(args)
    if args.replay_fixtures:
        args.replay_fixtures = args.replay_fixtures.resolve()
        replay(args, inputs, compiler)
    else:
        run_fresh(args, inputs)


if __name__ == "__main__":
    main()
