# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the STEP_APPROVALS extension (RFC 0200).

Covers Template Schemas §3 (``approval`` on ``<StepTemplate>``, approval-only
Steps), §3.7 ``<StepApproval>``, §3.7.1 ``<ApprovalMessage>``, §3.7.2
``<ApprovalLink>`` and §3.7.3 (Format String scope), and the resolution of an
approval's Format Strings at job creation. The conformance fixtures are run by
``test/openjd/model_v0/conformance``.
"""

from __future__ import annotations

import pickle
from typing import Any, Optional

import pytest
from pydantic import ValidationError

from openjd.model import (
    DecodeValidationError,
    ParameterValue,
    ParameterValueType,
    StepParameterSpaceIterator,
    create_job,
    decode_job_template,
    model_to_object,
)
from openjd.model.v2023_09 import (
    APPROVAL_EXPIRES_MAX_SECONDS,
    APPROVAL_EXPIRES_MIN_SECONDS,
    APPROVAL_LINK_LABEL_MAX_LENGTH,
    APPROVAL_LINKS_MAX,
    APPROVAL_MESSAGE_MAX_LENGTH,
    ApprovalLink,
    ApprovalLinkTemplate,
    ApprovalOnReject,
    ExtensionName,
    Job,
    JobTemplate,
    Step,
    StepApproval,
    StepApprovalTemplate,
    StepTemplate,
)

_ALL_EXTENSIONS = [e.value for e in ExtensionName]
_SCRIPT = {"actions": {"onRun": {"command": "echo"}}}
_MSG_PATTERN = "(?-m:^(?:[^\\u0000-\\u001F\\u007F-\\u009F]|[\\r\\n\\t])+\\Z)"
_URL_PATTERN = "(?-m:^[^\\u0000-\\u001F\\u007F-\\u009F]+\\Z)"


def _template(
    *steps: dict[str, Any],
    extensions: Optional[list[str]] = None,
    parameters: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    template: dict[str, Any] = {
        "specificationVersion": "jobtemplate-2023-09",
        "name": "TestJob",
        "steps": list(steps),
    }
    if extensions is None:
        extensions = ["STEP_APPROVALS"]
    if extensions:
        template["extensions"] = extensions
    if parameters:
        template["parameterDefinitions"] = parameters
    return template


def _review(approval: Any, **fields: Any) -> dict[str, Any]:
    return {"name": "Review", "approval": approval, **fields}


def _decode(template: dict[str, Any]) -> JobTemplate:
    return decode_job_template(template=template, supported_extensions=_ALL_EXTENSIONS)


def _decode_error(template: dict[str, Any]) -> str:
    with pytest.raises(DecodeValidationError) as excinfo:
        _decode(template)
    return str(excinfo.value)


def _one_error(loc: str, message: str) -> str:
    return f"1 validation errors for JobTemplate\n{loc}:\n\t{message}"


def _create(template: dict[str, Any], values: Optional[dict[str, ParameterValue]] = None) -> Job:
    return create_job(job_template=_decode(template), job_parameter_values=values or {})


def _create_error(template: dict[str, Any], values: Optional[dict[str, ParameterValue]]) -> str:
    job_template = _decode(template)
    with pytest.raises(DecodeValidationError) as excinfo:
        create_job(job_template=job_template, job_parameter_values=values or {})
    return str(excinfo.value)


def _string(value: str) -> ParameterValue:
    return ParameterValue(type=ParameterValueType.STRING, value=value)


def _path(value: str) -> ParameterValue:
    return ParameterValue(type=ParameterValueType.PATH, value=value)


_STRING_PARAM = {"name": "Text", "type": "STRING"}
_PATH_PARAM = {"name": "OutputDir", "type": "PATH"}


# ==================================================================
# §3: the `approval` property, the extension, and approval-only Steps
# ==================================================================


# An unsupported extension is not enabled, so the approval is also reported.
_UNSUPPORTED = (
    "2 validation errors for JobTemplate\n"
    "extensions:\n\tUnsupported extension names: STEP_APPROVALS\n"
    "steps[0]:\n\tThe 'approval' field requires the STEP_APPROVALS extension."
)


class TestExtensionGating:
    def test_extension_name(self) -> None:
        assert ExtensionName.STEP_APPROVALS.value == "STEP_APPROVALS"

    def test_approval_requires_the_extension(self) -> None:
        template = _template(_review({"message": "Approve?"}, script=_SCRIPT), extensions=[])

        assert _decode_error(template) == _one_error(
            "steps[0]", "The 'approval' field requires the STEP_APPROVALS extension."
        )

    def test_approval_only_step_requires_the_extension(self) -> None:
        template = _template(_review({"message": "Approve?"}), extensions=["EXPR"])

        assert _decode_error(template) == _one_error(
            "steps[0]", "The 'approval' field requires the STEP_APPROVALS extension."
        )

    def test_the_caller_must_support_the_extension(self) -> None:
        template = _template(_review({"message": "Approve?"}))

        with pytest.raises(DecodeValidationError) as excinfo:
            decode_job_template(template=template, supported_extensions=["EXPR"])

        assert str(excinfo.value) == _UNSUPPORTED

    def test_by_default_no_extension_is_supported(self) -> None:
        with pytest.raises(DecodeValidationError) as excinfo:
            decode_job_template(template=_template(_review({"message": "Approve?"})))

        assert str(excinfo.value) == _UNSUPPORTED

    def test_without_the_extension_a_step_still_needs_a_script(self) -> None:
        # The message is unchanged when the extension is not listed.
        template = _template({"name": "Step1"}, extensions=[])

        assert _decode_error(template) == _one_error(
            "steps[0]",
            "Must specify either 'script' or a script interpreter (python, bash, cmd, powershell, node).",
        )

    def test_without_the_extension_templates_are_unchanged(self) -> None:
        job_template = _decode(_template({"name": "Step1", "script": _SCRIPT}, extensions=[]))
        job = create_job(job_template=job_template, job_parameter_values={})

        assert job_template.steps[0].approval is None
        assert not job_template.steps[0].is_approval_only
        assert job.steps[0].approval is None
        assert job.steps[0].script is not None
        assert not job.steps[0].is_approval_only
        assert "approval" not in model_to_object(model=job)["steps"][0]

    def test_explicit_null_approval_is_absent(self) -> None:
        template = _template(_review(None, script=_SCRIPT), extensions=[])

        assert _decode(template).steps[0].approval is None


class TestScriptOrApproval:
    def test_neither_script_nor_approval(self) -> None:
        template = _template({"name": "Step1"})

        assert _decode_error(template) == _one_error(
            "steps[0]",
            "Must specify 'script', a script interpreter (python, bash, cmd, powershell, node), or 'approval'.",
        )

    def test_null_approval_is_not_an_approval(self) -> None:
        template = _template(_review(None))

        assert _decode_error(template) == _one_error(
            "steps[0]",
            "Must specify 'script', a script interpreter (python, bash, cmd, powershell, node), or 'approval'.",
        )

    def test_approval_with_script(self) -> None:
        step = _decode(_template(_review({"message": "Approve?"}, script=_SCRIPT))).steps[0]

        assert step.approval is not None
        assert step.script is not None
        assert not step.is_approval_only

    def test_approval_with_simple_action(self) -> None:
        template = _template(
            _review({"message": "Approve?"}, bash={"script": "echo hello"}),
            extensions=["STEP_APPROVALS", "FEATURE_BUNDLE_1"],
        )

        step = _decode(template).steps[0]

        assert step.approval is not None
        assert step.bash is not None
        assert not step.is_approval_only

    def test_simple_action_still_requires_feature_bundle_1(self) -> None:
        template = _template(_review({"message": "Approve?"}, bash={"script": "echo hello"}))

        assert _decode_error(template) == _one_error(
            "steps[0]",
            "Script interpreter syntax sugar (python, bash, cmd, powershell, node) requires the FEATURE_BUNDLE_1 extension.",
        )

    def test_approval_only_step(self) -> None:
        step = _decode(_template(_review({"message": "Approve?"}))).steps[0]

        assert step.script is None
        assert step.is_approval_only

    def test_approval_only_step_with_description_and_dependencies(self) -> None:
        template = _template(
            {"name": "Preview", "script": _SCRIPT},
            _review(
                {"message": "Approve?"},
                description="Review the preview",
                dependencies=[{"dependsOn": "Preview"}],
            ),
            {"name": "Final", "script": _SCRIPT, "dependencies": [{"dependsOn": "Review"}]},
        )

        steps = _decode(template).steps

        assert [s.is_approval_only for s in steps] == [False, True, False]


class TestApprovalOnlyStepRestrictions:
    @pytest.mark.parametrize(
        "field, value, extensions",
        [
            pytest.param(
                "stepEnvironments",
                [{"name": "Env1", "variables": {"FOO": "bar"}}],
                ["STEP_APPROVALS"],
                id="stepEnvironments",
            ),
            pytest.param(
                "hostRequirements",
                {"attributes": [{"name": "attr.worker.os.family", "anyOf": ["linux"]}]},
                ["STEP_APPROVALS"],
                id="hostRequirements",
            ),
            pytest.param(
                "parameterSpace",
                {"taskParameterDefinitions": [{"name": "F", "type": "INT", "range": "1-3"}]},
                ["STEP_APPROVALS"],
                id="parameterSpace",
            ),
            pytest.param("let", ["x = 1"], ["STEP_APPROVALS", "EXPR"], id="let"),
        ],
    )
    def test_disallowed_field(self, field: str, value: Any, extensions: list[str]) -> None:
        template = _template(
            _review({"message": "Approve?"}, **{field: value}), extensions=extensions
        )

        assert _decode_error(template) == _one_error(
            "steps[0]",
            "An approval-only Step (one with 'approval' and no 'script' or script interpreter) "
            f"must not have: {field}.",
        )

    def test_all_disallowed_fields_are_reported(self) -> None:
        template = _template(
            _review(
                {"message": "Approve?"},
                let=["x = 1"],
                parameterSpace={
                    "taskParameterDefinitions": [{"name": "F", "type": "INT", "range": "1-3"}]
                },
                hostRequirements={"amounts": [{"name": "amount.worker.vcpu", "min": 1}]},
                stepEnvironments=[{"name": "Env1", "variables": {"FOO": "bar"}}],
            ),
            extensions=["STEP_APPROVALS", "EXPR"],
        )

        assert _decode_error(template) == _one_error(
            "steps[0]",
            "An approval-only Step (one with 'approval' and no 'script' or script interpreter) "
            "must not have: stepEnvironments, hostRequirements, parameterSpace, let.",
        )

    def test_a_step_with_a_script_may_have_them(self) -> None:
        template = _template(
            _review(
                {"message": "Approve?"},
                script=_SCRIPT,
                let=["x = 1"],
                parameterSpace={
                    "taskParameterDefinitions": [{"name": "F", "type": "INT", "range": "1-3"}]
                },
                hostRequirements={"amounts": [{"name": "amount.worker.vcpu", "min": 1}]},
                stepEnvironments=[{"name": "Env1", "variables": {"FOO": "bar"}}],
            ),
            extensions=["STEP_APPROVALS", "EXPR"],
        )

        _decode(template)


# ==================================================================
# §3.7 <StepApproval>
# ==================================================================


class TestStepApprovalFields:
    def test_all_fields(self) -> None:
        approval = (
            _decode(
                _template(
                    _review(
                        {
                            "message": "Approve?",
                            "links": [{"label": "Docs", "url": "https://example.com/docs"}],
                            "expiresAfterSeconds": 600,
                            "onReject": "FAIL",
                        }
                    )
                )
            )
            .steps[0]
            .approval
        )

        assert isinstance(approval, StepApprovalTemplate)
        assert approval.message == "Approve?"
        assert approval.links is not None
        assert isinstance(approval.links[0], ApprovalLinkTemplate)
        assert approval.expiresAfterSeconds == 600
        assert approval.onReject == ApprovalOnReject.FAIL

    def test_optional_fields_default_to_none(self) -> None:
        approval = _decode(_template(_review({"message": "Approve?"}))).steps[0].approval

        assert approval is not None
        assert approval.links is None
        assert approval.expiresAfterSeconds is None
        assert approval.onReject is None

    def test_empty_approval(self) -> None:
        assert _decode_error(_template(_review({}))) == _one_error(
            "steps[0] -> approval -> message", "Field required"
        )

    def test_unknown_field(self) -> None:
        template = _template(_review({"message": "Approve?", "notAField": True}))

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> notAField", "Extra inputs are not permitted"
        )

    def test_not_an_object(self) -> None:
        assert _decode_error(_template(_review("Approve?"))) == _one_error(
            "steps[0] -> approval",
            "Input should be a valid dictionary or instance of StepApprovalTemplate",
        )


class TestExpiresAfterSeconds:
    @pytest.mark.parametrize(
        "value", [APPROVAL_EXPIRES_MIN_SECONDS, 3600, APPROVAL_EXPIRES_MAX_SECONDS]
    )
    def test_in_range(self, value: int) -> None:
        template = _template(_review({"message": "Approve?", "expiresAfterSeconds": value}))

        approval = _decode(template).steps[0].approval

        assert approval is not None
        assert approval.expiresAfterSeconds == value

    def test_limits(self) -> None:
        assert (APPROVAL_EXPIRES_MIN_SECONDS, APPROVAL_EXPIRES_MAX_SECONDS) == (60, 2592000)

    @pytest.mark.parametrize(
        "value, message",
        [
            pytest.param(59, "Input should be greater than or equal to 60", id="below min"),
            pytest.param(2592001, "Input should be less than or equal to 2592000", id="above max"),
            pytest.param(600.5, "Input should be a valid integer", id="float"),
            pytest.param(600.0, "Input should be a valid integer", id="integral float"),
            pytest.param("600", "Input should be a valid integer", id="string"),
            pytest.param("{{Param.Text}}", "Input should be a valid integer", id="format string"),
            pytest.param(True, "Input should be a valid integer", id="bool"),
        ],
    )
    def test_invalid(self, value: Any, message: str) -> None:
        template = _template(_review({"message": "Approve?", "expiresAfterSeconds": value}))

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> expiresAfterSeconds", message
        )


class TestOnReject:
    def test_fail(self) -> None:
        approval = (
            _decode(_template(_review({"message": "M", "onReject": "FAIL"}))).steps[0].approval
        )

        assert approval is not None
        assert approval.onReject is ApprovalOnReject.FAIL

    @pytest.mark.parametrize("value", ["SKIP_DEPENDENTS", "fail", ""])
    def test_only_fail(self, value: str) -> None:
        template = _template(_review({"message": "Approve?", "onReject": value}))

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> onReject", "Input should be 'FAIL'"
        )


# ==================================================================
# §3.7.1 <ApprovalMessage>
# ==================================================================


class TestApprovalMessage:
    @pytest.mark.parametrize(
        "message",
        [
            pytest.param("x", id="one character"),
            pytest.param("a" * APPROVAL_MESSAGE_MAX_LENGTH, id="max length"),
            pytest.param("Line 1\nLine 2\r\n\tIndented", id="newline, CR and tab"),
            pytest.param("Prüfen Sie die Vorschau 🎬", id="unicode"),
            pytest.param("{{Param.Text}}" + "a" * APPROVAL_MESSAGE_MAX_LENGTH, id="unresolved"),
        ],
    )
    def test_valid(self, message: str) -> None:
        template = _template(_review({"message": message}), parameters=[_STRING_PARAM])

        approval = _decode(template).steps[0].approval

        assert approval is not None
        assert approval.message == message

    def test_max_length(self) -> None:
        assert APPROVAL_MESSAGE_MAX_LENGTH == 2048

    def test_missing(self) -> None:
        template = _template(_review({"expiresAfterSeconds": 600}))

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> message", "Field required"
        )

    def test_empty(self) -> None:
        assert _decode_error(_template(_review({"message": ""}))) == _one_error(
            "steps[0] -> approval -> message", "String must be at least 1 characters long"
        )

    def test_too_long(self) -> None:
        template = _template(_review({"message": "a" * (APPROVAL_MESSAGE_MAX_LENGTH + 1)}))

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> message", "message must be at most 2048 characters long"
        )

    @pytest.mark.parametrize("char", ["\x00", "\x07", "\x1b", "\x7f", "\x9f"])
    def test_control_characters(self, char: str) -> None:
        template = _template(_review({"message": f"Approve{char}?"}))

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> message",
            f"String does not match the required pattern: {_MSG_PATTERN}",
        )

    def test_not_a_string(self) -> None:
        assert _decode_error(_template(_review({"message": 12}))) == _one_error(
            "steps[0] -> approval -> message", "String required"
        )

    def test_malformed_format_string(self) -> None:
        template = _template(_review({"message": "Approve {{ Param.Text"}))

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> message",
            "Failed to parse interpolation expression at [0, 21]. Reason: Braces mismatch.",
        )


# ==================================================================
# §3.7.2 <ApprovalLink>
# ==================================================================


class TestApprovalLinks:
    def test_max_links(self) -> None:
        links = [{"label": f"L{i}", "url": f"https://example.com/{i}"} for i in range(10)]

        approval = _decode(_template(_review({"message": "M", "links": links}))).steps[0].approval

        assert approval is not None and approval.links is not None
        assert len(approval.links) == APPROVAL_LINKS_MAX == 10

    def test_too_many_links(self) -> None:
        links = [{"label": f"L{i}", "url": f"https://example.com/{i}"} for i in range(11)]

        assert _decode_error(_template(_review({"message": "M", "links": links}))) == _one_error(
            "steps[0] -> approval -> links",
            "List should have at most 10 items after validation, not 11",
        )

    def test_empty_links(self) -> None:
        assert _decode_error(_template(_review({"message": "M", "links": []}))) == _one_error(
            "steps[0] -> approval -> links",
            "List should have at least 1 item after validation, not 0",
        )

    def test_unknown_field(self) -> None:
        links = [{"label": "L", "url": "https://example.com", "target": "_blank"}]

        assert _decode_error(_template(_review({"message": "M", "links": links}))) == _one_error(
            "steps[0] -> approval -> links[0] -> target", "Extra inputs are not permitted"
        )


class TestApprovalLinkLabel:
    def test_max_length(self) -> None:
        label = "a" * APPROVAL_LINK_LABEL_MAX_LENGTH
        links = [{"label": label, "url": "https://example.com"}]

        approval = _decode(_template(_review({"message": "M", "links": links}))).steps[0].approval

        assert approval is not None and approval.links is not None
        assert approval.links[0].label == label
        assert APPROVAL_LINK_LABEL_MAX_LENGTH == 64

    @pytest.mark.parametrize(
        "label, message",
        [
            pytest.param("a" * 65, "String should have at most 64 characters", id="too long"),
            pytest.param("", "String should have at least 1 character", id="empty"),
            pytest.param(
                "a\nb",
                "String should match pattern '(?-m:^[^\\u0000-\\u001F\\u007F-\\u009F]+\\z)'",
                id="newline",
            ),
            pytest.param(12, "Input should be a valid string", id="not a string"),
        ],
    )
    def test_invalid(self, label: Any, message: str) -> None:
        links = [{"label": label, "url": "https://example.com"}]

        assert _decode_error(_template(_review({"message": "M", "links": links}))) == _one_error(
            "steps[0] -> approval -> links[0] -> label", message
        )

    def test_missing(self) -> None:
        links = [{"url": "https://example.com"}]

        assert _decode_error(_template(_review({"message": "M", "links": links}))) == _one_error(
            "steps[0] -> approval -> links[0] -> label", "Field required"
        )

    def test_label_is_not_a_format_string(self) -> None:
        links = [{"label": "{{Param.Text}}", "url": "https://example.com"}]
        template = _template(_review({"message": "M", "links": links}))

        approval = _create(template).steps[0].approval

        assert approval is not None and approval.links is not None
        assert approval.links[0].label == "{{Param.Text}}"


class TestApprovalLinkTarget:
    @pytest.mark.parametrize(
        "link, message",
        [
            pytest.param(
                {"label": "Neither"},
                "A link must provide exactly one of 'url' or 'parameter'.",
                id="neither",
            ),
            pytest.param(
                {"label": "Both", "url": "https://example.com", "parameter": "OutputDir"},
                "A link must provide exactly one of 'url' or 'parameter', not both.",
                id="both",
            ),
            pytest.param(
                {"label": "Url", "url": "https://example.com", "subpath": "preview"},
                "'subpath' may only be provided with 'parameter'.",
                id="subpath with url",
            ),
            pytest.param(
                {"label": "Subpath", "subpath": "preview"},
                "A link must provide exactly one of 'url' or 'parameter'.",
                id="subpath alone",
            ),
        ],
    )
    def test_exactly_one_target(self, link: dict[str, Any], message: str) -> None:
        template = _template(_review({"message": "M", "links": [link]}), parameters=[_PATH_PARAM])

        assert _decode_error(template) == _one_error("steps[0] -> approval -> links[0]", message)

    def test_link_is_not_an_object(self) -> None:
        template = _template(_review({"message": "M", "links": ["https://example.com"]}))

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> links[0]",
            "Input should be a valid dictionary or instance of ApprovalLinkTemplate",
        )


class TestApprovalLinkUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com",
            "http://example.com/shots/sh010?tab=review#frames",
            "HTTPS://EXAMPLE.COM/",
            "https://user@review.example.com:8443/path",
            "https://[::1]/review",
            "https://review.example.com/{{Param.Text}}",
            "{{Param.Text}}",
        ],
    )
    def test_valid(self, url: str) -> None:
        template = _template(
            _review({"message": "M", "links": [{"label": "L", "url": url}]}),
            parameters=[_STRING_PARAM],
        )

        approval = _decode(template).steps[0].approval

        assert approval is not None and approval.links is not None
        assert approval.links[0].url == url

    @pytest.mark.parametrize(
        "url",
        [
            pytest.param("javascript:alert(1)", id="javascript"),
            pytest.param("ftp://example.com/file", id="ftp"),
            pytest.param("file:///etc/passwd", id="file"),
            pytest.param("mailto:someone@example.com", id="mailto"),
            pytest.param("/shots/sh010", id="relative"),
            pytest.param("example.com/shots", id="no scheme"),
            pytest.param("//example.com/shots", id="scheme relative"),
            pytest.param("https://", id="no host"),
            pytest.param("https:///path", id="empty host"),
            pytest.param("https://exa mple.com", id="space"),
            pytest.param("https://[::1/review", id="bad ipv6"),
        ],
    )
    def test_invalid_literal(self, url: str) -> None:
        template = _template(_review({"message": "M", "links": [{"label": "L", "url": url}]}))

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> links[0] -> url",
            f"url must be an absolute URI whose scheme is http or https, not {url!r}.",
        )

    def test_empty(self) -> None:
        template = _template(_review({"message": "M", "links": [{"label": "L", "url": ""}]}))

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> links[0] -> url", "String must be at least 1 characters long"
        )

    def test_control_character(self) -> None:
        template = _template(
            _review({"message": "M", "links": [{"label": "L", "url": "https://e.com/\x07"}]})
        )

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> links[0] -> url",
            f"String does not match the required pattern: {_URL_PATTERN}",
        )


class TestApprovalLinkParameter:
    def test_path_parameter(self) -> None:
        template = _template(
            _review({"message": "M", "links": [{"label": "Out", "parameter": "OutputDir"}]}),
            parameters=[_PATH_PARAM],
        )

        approval = _decode(template).steps[0].approval

        assert approval is not None and approval.links is not None
        assert approval.links[0].parameter == "OutputDir"

    def test_undefined(self) -> None:
        template = _template(
            _review({"message": "M", "links": [{"label": "L", "parameter": "NotDefined"}]})
        )

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> links[0] -> parameter",
            "Job parameter 'NotDefined' is not defined in the Job Template.",
        )

    @pytest.mark.parametrize(
        "definition, type_name, extensions",
        [
            pytest.param({"type": "STRING"}, "STRING", ["STEP_APPROVALS"], id="STRING"),
            pytest.param({"type": "INT"}, "INT", ["STEP_APPROVALS"], id="INT"),
            pytest.param({"type": "FLOAT"}, "FLOAT", ["STEP_APPROVALS"], id="FLOAT"),
            pytest.param(
                {"type": "LIST[PATH]"}, "LIST[PATH]", ["STEP_APPROVALS", "EXPR"], id="LIST[PATH]"
            ),
        ],
    )
    def test_not_path(
        self, definition: dict[str, Any], type_name: str, extensions: list[str]
    ) -> None:
        template = _template(
            _review({"message": "M", "links": [{"label": "L", "parameter": "P"}]}),
            parameters=[{"name": "P", **definition}],
            extensions=extensions,
        )

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> links[0] -> parameter",
            f"Job parameter 'P' must be of type PATH to be the target of an approval link; "
            f"it is of type {type_name}.",
        )

    def test_errors_name_each_link(self) -> None:
        template = _template(
            {"name": "A", "script": _SCRIPT},
            {
                "name": "B",
                "approval": {
                    "message": "M",
                    "links": [
                        {"label": "Ok", "parameter": "OutputDir"},
                        {"label": "Missing", "parameter": "Missing"},
                        {"label": "Text", "parameter": "Text"},
                    ],
                },
            },
            parameters=[_PATH_PARAM, _STRING_PARAM],
        )

        assert _decode_error(template) == (
            "2 validation errors for JobTemplate\n"
            "steps[1] -> approval -> links[1] -> parameter:\n"
            "\tJob parameter 'Missing' is not defined in the Job Template.\n"
            "steps[1] -> approval -> links[2] -> parameter:\n"
            "\tJob parameter 'Text' must be of type PATH to be the target of an approval link; "
            "it is of type STRING."
        )

    def test_not_an_identifier(self) -> None:
        template = _template(
            _review({"message": "M", "links": [{"label": "L", "parameter": "Output Dir"}]})
        )

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> links[0] -> parameter",
            "String should match pattern '(?-m:^[A-Za-z_][A-Za-z0-9_]*\\z)'",
        )


class TestApprovalLinkSubpath:
    @pytest.mark.parametrize(
        "subpath",
        ["preview", "preview/frames", "preview\\frames", "...", "a..b/c", ".", "{{Param.Text}}"],
    )
    def test_valid(self, subpath: str) -> None:
        template = _template(
            _review(
                {
                    "message": "M",
                    "links": [{"label": "L", "parameter": "OutputDir", "subpath": subpath}],
                }
            ),
            parameters=[_PATH_PARAM, _STRING_PARAM],
        )

        approval = _decode(template).steps[0].approval

        assert approval is not None and approval.links is not None
        assert approval.links[0].subpath == subpath

    @pytest.mark.parametrize(
        "subpath, message",
        [
            pytest.param(
                "/etc/passwd", "subpath must be a relative path, not '/etc/passwd'.", id="posix"
            ),
            pytest.param(
                "C:\\Windows", "subpath must be a relative path, not 'C:\\\\Windows'.", id="drive"
            ),
            pytest.param(
                "C:relative", "subpath must be a relative path, not 'C:relative'.", id="drive rel"
            ),
            pytest.param(
                "\\\\server\\share",
                "subpath must be a relative path, not '\\\\\\\\server\\\\share'.",
                id="unc",
            ),
            pytest.param(
                "\\rooted", "subpath must be a relative path, not '\\\\rooted'.", id="root"
            ),
            pytest.param("..", "subpath must not contain a '..' path segment: '..'.", id="parent"),
            pytest.param(
                "preview/../../secret",
                "subpath must not contain a '..' path segment: 'preview/../../secret'.",
                id="posix parent",
            ),
            pytest.param(
                "preview\\..\\secret",
                "subpath must not contain a '..' path segment: 'preview\\\\..\\\\secret'.",
                id="windows parent",
            ),
            pytest.param(
                "preview/..",
                "subpath must not contain a '..' path segment: 'preview/..'.",
                id="trailing parent",
            ),
        ],
    )
    def test_invalid_literal(self, subpath: str, message: str) -> None:
        template = _template(
            _review(
                {
                    "message": "M",
                    "links": [{"label": "L", "parameter": "OutputDir", "subpath": subpath}],
                }
            ),
            parameters=[_PATH_PARAM],
        )

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> links[0] -> subpath", message
        )


# ==================================================================
# §3.7.3 Format String scope
# ==================================================================


class TestFormatStringScope:
    @pytest.mark.parametrize(
        "reference",
        ["{{Param.Text}}", "{{RawParam.Text}}", "{{RawParam.OutputDir}}"],
    )
    def test_in_scope(self, reference: str) -> None:
        template = _template(
            _review(
                {
                    "message": reference,
                    "links": [
                        {"label": "U", "url": f"https://e.com/{reference}"},
                        {"label": "P", "parameter": "OutputDir", "subpath": reference},
                    ],
                }
            ),
            parameters=[_STRING_PARAM, _PATH_PARAM],
        )

        _decode(template)

    def test_path_param_is_only_raw(self) -> None:
        template = _template(_review({"message": "{{Param.OutputDir}}"}), parameters=[_PATH_PARAM])

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> message",
            "Variable Param.OutputDir does not exist at this location. Did you mean: RawParam.OutputDir",
        )

    @pytest.mark.parametrize(
        "field, loc",
        [
            pytest.param("message", "steps[0] -> approval -> message", id="message"),
            pytest.param("url", "steps[0] -> approval -> links[0] -> url", id="url"),
            pytest.param("subpath", "steps[0] -> approval -> links[0] -> subpath", id="subpath"),
        ],
    )
    @pytest.mark.parametrize(
        "variable",
        ["Task.Param.Frame", "Session.WorkingDirectory", "Param.Undefined"],
    )
    def test_out_of_scope(self, field: str, loc: str, variable: str) -> None:
        approval: dict[str, Any] = {"message": "M"}
        if field == "message":
            approval["message"] = f"{{{{{variable}}}}}"
        elif field == "url":
            approval["links"] = [{"label": "L", "url": f"https://e.com/{{{{{variable}}}}}"}]
        else:
            approval["links"] = [
                {"label": "L", "parameter": "OutputDir", "subpath": f"{{{{{variable}}}}}"}
            ]
        template = _template(_review(approval), parameters=[_PATH_PARAM])

        assert _decode_error(template) == _one_error(
            loc, f"Variable {variable} does not exist at this location."
        )

    def test_task_params_of_the_step_are_not_in_scope(self) -> None:
        template = _template(
            _review(
                {"message": "Frame {{Task.Param.Frame}}"},
                script=_SCRIPT,
                parameterSpace={
                    "taskParameterDefinitions": [{"name": "Frame", "type": "INT", "range": "1-3"}]
                },
            )
        )

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> message",
            "Variable Task.Param.Frame does not exist at this location.",
        )

    def test_job_and_step_name_require_expr(self) -> None:
        template = _template(_review({"message": "{{Job.Name}} {{Step.Name}}"}))

        assert _decode_error(template) == (
            "2 validation errors for JobTemplate\n"
            "steps[0] -> approval -> message:\n"
            "\tVariable Job.Name does not exist at this location.\n"
            "steps[0] -> approval -> message:\n"
            "\tVariable Step.Name does not exist at this location."
        )

    def test_job_and_step_name_with_expr(self) -> None:
        template = _template(
            _review(
                {
                    "message": "{{Job.Name}} {{Step.Name}}",
                    "links": [
                        {"label": "L", "url": "https://e.com/{{Job.Name}}/{{Step.Name}}"},
                        {"label": "P", "parameter": "OutputDir", "subpath": "{{Step.Name}}"},
                    ],
                }
            ),
            extensions=["STEP_APPROVALS", "EXPR"],
            parameters=[_PATH_PARAM],
        )

        _decode(template)

    def test_let_bindings_are_not_in_scope(self) -> None:
        template = _template(
            _review(
                {
                    "message": "{{shot}}",
                    "links": [
                        {"label": "L", "url": "https://e.com/{{shot}}"},
                        {"label": "P", "parameter": "OutputDir", "subpath": "{{shot}}"},
                    ],
                },
                script=_SCRIPT,
                let=["shot = 'sh010'"],
            ),
            extensions=["STEP_APPROVALS", "EXPR"],
            parameters=[_PATH_PARAM],
        )

        assert _decode_error(template) == (
            "3 validation errors for JobTemplate\n"
            "steps[0] -> approval -> message:\n"
            "\tVariable shot does not exist at this location.\n"
            "steps[0] -> approval -> links[0] -> url:\n"
            "\tVariable shot does not exist at this location.\n"
            "steps[0] -> approval -> links[1] -> subpath:\n"
            "\tVariable shot does not exist at this location."
        )

    def test_expr_expressions(self) -> None:
        template = _template(
            _review({"message": "{{ Param.Text.upper() }} x{{ Param.Count * 2 }}"}),
            extensions=["STEP_APPROVALS", "EXPR"],
            parameters=[_STRING_PARAM, {"name": "Count", "type": "INT"}],
        )
        values = {
            "Text": _string("sh010"),
            "Count": ParameterValue(type=ParameterValueType.INT, value="3"),
        }

        approval = _create(template, values).steps[0].approval

        assert approval is not None
        assert approval.message == "SH010 x6"

    def test_expr_type_errors_are_reported(self) -> None:
        template = _template(
            _review({"message": "{{ Param.Count + 'a' }}"}),
            extensions=["STEP_APPROVALS", "EXPR"],
            parameters=[{"name": "Count", "type": "INT"}],
        )

        assert _decode_error(template) == _one_error(
            "steps[0] -> approval -> message",
            "Cannot use '+' operator with int and string\n"
            "  Param.Count + 'a'\n"
            "  ~~~~~~~~~~~~^~~~~",
        )


# ==================================================================
# Job creation: resolved approvals
# ==================================================================


class TestCreateJob:
    def test_resolves_message_and_links(self) -> None:
        template = _template(
            {"name": "Preview", "script": _SCRIPT},
            _review(
                {
                    "message": "Check {{Param.Text}} in {{RawParam.OutputDir}}",
                    "links": [
                        {"label": "Frames", "parameter": "OutputDir", "subpath": "{{Param.Text}}"},
                        {"label": "Dir", "parameter": "OutputDir"},
                        {"label": "Page", "url": "https://review.example.com/{{Param.Text}}"},
                    ],
                    "expiresAfterSeconds": 259200,
                    "onReject": "FAIL",
                },
                dependencies=[{"dependsOn": "Preview"}],
            ),
            parameters=[_STRING_PARAM, _PATH_PARAM],
        )
        values = {"Text": _string("sh010"), "OutputDir": _path("/mnt/shots/out")}

        job = _create(template, values)
        step = job.steps[1]

        assert isinstance(step, Step)
        assert step.is_approval_only
        assert step.script is None
        assert step.approval == StepApproval(
            message="Check sh010 in /mnt/shots/out",
            links=[
                ApprovalLink(
                    label="Frames", parameter="OutputDir", path="/mnt/shots/out", subpath="sh010"
                ),
                ApprovalLink(label="Dir", parameter="OutputDir", path="/mnt/shots/out"),
                ApprovalLink(label="Page", url="https://review.example.com/sh010"),
            ],
            expiresAfterSeconds=259200,
            onReject=ApprovalOnReject.FAIL,
        )
        assert [link.target for link in step.approval.links or []] == [
            "/mnt/shots/out/sh010",
            "/mnt/shots/out",
            "https://review.example.com/sh010",
        ]
        assert job.steps[0].approval is None

    def test_path_values_are_shown_as_submitted(self) -> None:
        # No path mapping, and no normalization: the value is RawParam.*.
        template = _template(
            _review(
                {
                    "message": "{{RawParam.OutputDir}}",
                    "links": [{"label": "L", "parameter": "OutputDir"}],
                }
            ),
            parameters=[_PATH_PARAM],
        )

        approval = _create(template, {"OutputDir": _path("Z:\\Shots\\sh010")}).steps[0].approval

        assert approval is not None and approval.links is not None
        assert approval.message == "Z:\\Shots\\sh010"
        assert approval.links[0].path == "Z:\\Shots\\sh010"

    def test_path_default(self) -> None:
        template = _template(
            _review({"message": "M", "links": [{"label": "L", "parameter": "OutputDir"}]}),
            parameters=[{**_PATH_PARAM, "default": "/mnt/out"}],
        )

        approval = _create(template).steps[0].approval

        assert approval is not None and approval.links is not None
        assert approval.links[0].path == "/mnt/out"

    def test_job_and_step_name(self) -> None:
        template = _template(
            _review({"message": "{{Job.Name}}/{{Step.Name}}"}),
            extensions=["STEP_APPROVALS", "EXPR"],
        )
        template["name"] = "Shot {{Param.Text}}"
        template["parameterDefinitions"] = [_STRING_PARAM]

        approval = _create(template, {"Text": _string("sh010")}).steps[0].approval

        assert approval is not None
        assert approval.message == "Shot sh010/Review"

    def test_approval_with_script(self) -> None:
        template = _template(_review({"message": "Approve?"}, script=_SCRIPT))

        step = _create(template).steps[0]

        assert step.approval == StepApproval(message="Approve?")
        assert step.script is not None
        assert not step.is_approval_only

    def test_approval_with_simple_action_survives_desugaring(self) -> None:
        template = _template(
            _review({"message": "Approve {{Param.Text}}?"}, bash={"script": "echo hello"}),
            extensions=["STEP_APPROVALS", "FEATURE_BUNDLE_1"],
            parameters=[_STRING_PARAM],
        )

        step = _create(template, {"Text": _string("sh010")}).steps[0]

        assert step.approval == StepApproval(message="Approve sh010?")
        assert step.script is not None
        assert not step.is_approval_only

    def test_approval_only_step_has_one_task_with_no_parameters(self) -> None:
        step = _create(_template(_review({"message": "M"}))).steps[0]

        assert step.parameterSpace is None
        assert list(StepParameterSpaceIterator(space=step.parameterSpace)) == [{}]

    def test_message_too_long_once_resolved(self) -> None:
        template = _template(_review({"message": "a{{Param.Text}}"}), parameters=[_STRING_PARAM])

        message = _create_error(template, {"Text": _string("a" * APPROVAL_MESSAGE_MAX_LENGTH)})

        assert message == _one_error(
            "steps[0] -> approval -> message",
            "message must be at most 2048 characters long once its Format Strings are resolved; "
            "it is 2049 characters long",
        )

    def test_message_at_max_length_once_resolved(self) -> None:
        template = _template(_review({"message": "a{{Param.Text}}"}), parameters=[_STRING_PARAM])

        approval = _create(template, {"Text": _string("a" * 2047)}).steps[0].approval

        assert approval is not None
        assert len(approval.message) == APPROVAL_MESSAGE_MAX_LENGTH

    def test_message_empty_once_resolved(self) -> None:
        template = _template(_review({"message": "{{Param.Text}}"}), parameters=[_STRING_PARAM])

        assert _create_error(template, {"Text": _string("")}) == _one_error(
            "steps[0] -> approval -> message", "message must not be empty"
        )

    def test_message_control_character_once_resolved(self) -> None:
        template = _template(_review({"message": "{{Param.Text}}"}), parameters=[_STRING_PARAM])

        assert _create_error(template, {"Text": _string("a\x1bb")}) == _one_error(
            "steps[0] -> approval -> message",
            "message must not contain control characters other than newline, carriage return, "
            "and horizontal tab",
        )

    @pytest.mark.parametrize(
        "value",
        ["javascript:alert(1)", "ftp://example.com", "/relative", "https://", "https://a b/c"],
    )
    def test_url_invalid_once_resolved(self, value: str) -> None:
        template = _template(
            _review({"message": "M", "links": [{"label": "L", "url": "{{Param.Text}}"}]}),
            parameters=[_STRING_PARAM],
        )

        assert _create_error(template, {"Text": _string(value)}) == _one_error(
            "steps[0] -> approval -> links[0] -> url",
            f"url must be an absolute URI whose scheme is http or https, not {value!r}.",
        )

    @pytest.mark.parametrize(
        "value, message",
        [
            pytest.param("/etc", "subpath must be a relative path, not '/etc'.", id="absolute"),
            pytest.param(
                "a/../b", "subpath must not contain a '..' path segment: 'a/../b'.", id="parent"
            ),
        ],
    )
    def test_subpath_invalid_once_resolved(self, value: str, message: str) -> None:
        template = _template(
            _review(
                {
                    "message": "M",
                    "links": [
                        {"label": "L", "parameter": "OutputDir", "subpath": "{{Param.Text}}"}
                    ],
                }
            ),
            parameters=[_STRING_PARAM, _PATH_PARAM],
        )

        error = _create_error(template, {"Text": _string(value), "OutputDir": _path("/out")})

        assert error == _one_error("steps[0] -> approval -> links[0] -> subpath", message)

    def test_job_round_trips_through_model_to_object(self) -> None:
        template = _template(
            _review(
                {
                    "message": "M",
                    "links": [
                        {"label": "P", "parameter": "OutputDir", "subpath": "x"},
                        {"label": "U", "url": "https://e.com"},
                    ],
                    "expiresAfterSeconds": 60,
                }
            ),
            parameters=[_PATH_PARAM],
        )
        job = _create(template, {"OutputDir": _path("/out")})

        as_object = model_to_object(model=job)

        assert as_object["steps"][0] == {
            "name": "Review",
            "approval": {
                "message": "M",
                "links": [
                    {"label": "P", "parameter": "OutputDir", "path": "/out", "subpath": "x"},
                    {"label": "U", "url": "https://e.com"},
                ],
                "expiresAfterSeconds": 60,
            },
        }
        assert Job.model_validate(job.model_dump()) == job

    def test_template_round_trips_through_model_to_object(self) -> None:
        template = _template(
            _review(
                {
                    "message": "M {{Param.Text}}",
                    "links": [{"label": "P", "parameter": "OutputDir", "subpath": "x"}],
                    "expiresAfterSeconds": 60,
                    "onReject": "FAIL",
                }
            ),
            parameters=[_STRING_PARAM, _PATH_PARAM],
        )
        job_template = _decode(template)

        as_object = model_to_object(model=job_template)

        assert as_object["steps"] == template["steps"]
        assert _decode(as_object) == job_template

    def test_pickle(self) -> None:
        template = _template(
            _review(
                {
                    "message": "M {{Param.Text}}",
                    "links": [
                        {"label": "P", "parameter": "OutputDir", "subpath": "{{Param.Text}}"}
                    ],
                }
            ),
            parameters=[_STRING_PARAM, _PATH_PARAM],
        )
        job_template = _decode(template)
        job = create_job(
            job_template=job_template,
            job_parameter_values={"Text": _string("t"), "OutputDir": _path("/o")},
        )

        assert pickle.loads(pickle.dumps(job_template)) == job_template
        assert pickle.loads(pickle.dumps(job)) == job


# ==================================================================
# The Job's models
# ==================================================================


class TestStepModel:
    def test_requires_script_or_approval(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Step(name="S")

        assert excinfo.value.errors()[0]["msg"] == (
            "Value error, A Step must have a 'script', an 'approval', or both."
        )

    def test_approval_only(self) -> None:
        step = Step(name="S", approval=StepApproval(message="M"))

        assert step.is_approval_only


class TestApprovalLinkModel:
    @pytest.mark.parametrize(
        "path, subpath, target",
        [
            pytest.param("/mnt/out", "preview", "/mnt/out/preview", id="posix"),
            pytest.param("/mnt/out/", "preview", "/mnt/out/preview", id="posix trailing"),
            pytest.param("/", "preview", "/preview", id="posix root"),
            pytest.param("C:\\out", "preview", "C:\\out\\preview", id="windows"),
            pytest.param("C:\\out\\", "a\\b", "C:\\out\\a\\b", id="windows trailing"),
            pytest.param("C:/out", "preview", "C:/out/preview", id="windows forward"),
            pytest.param("/mnt/out", None, "/mnt/out", id="no subpath"),
            pytest.param("/mnt/out", "", "/mnt/out", id="empty subpath"),
        ],
    )
    def test_target(self, path: str, subpath: Optional[str], target: str) -> None:
        link = ApprovalLink(label="L", parameter="P", path=path, subpath=subpath)

        assert link.target == target

    def test_url_target(self) -> None:
        assert ApprovalLink(label="L", url="https://e.com").target == "https://e.com"

    @pytest.mark.parametrize(
        "fields, message",
        [
            pytest.param(
                {"parameter": "P"},
                "Value error, A link's 'path' is provided if and only if its 'parameter' is.",
                id="parameter without path",
            ),
            pytest.param(
                {"url": "https://e.com", "path": "/o"},
                "Value error, A link's 'path' is provided if and only if its 'parameter' is.",
                id="path without parameter",
            ),
            pytest.param(
                {},
                "Value error, A link must provide exactly one of 'url' or 'parameter'.",
                id="no target",
            ),
            pytest.param(
                {"url": "https://e.com", "parameter": "P", "path": "/o"},
                "Value error, A link must provide exactly one of 'url' or 'parameter', not both.",
                id="both",
            ),
        ],
    )
    def test_invalid(self, fields: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ApprovalLink(label="L", **fields)

        assert excinfo.value.errors()[0]["msg"] == message

    def test_url_is_validated(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ApprovalLink(label="L", url="javascript:alert(1)")

        assert excinfo.value.errors()[0]["msg"] == (
            "Value error, url must be an absolute URI whose scheme is http or https, "
            "not 'javascript:alert(1)'."
        )

    def test_subpath_is_validated(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            ApprovalLink(label="L", parameter="P", path="/o", subpath="../x")

        assert excinfo.value.errors()[0]["msg"] == (
            "Value error, subpath must not contain a '..' path segment: '../x'."
        )


class TestStepTemplateModel:
    def test_is_approval_only(self) -> None:
        job_template = _decode(
            _template(
                _review({"message": "M"}),
                {"name": "WithScript", "script": _SCRIPT, "approval": {"message": "M"}},
                {"name": "Plain", "script": _SCRIPT},
                {"name": "Sugar", "bash": {"script": "echo"}, "approval": {"message": "M"}},
                extensions=["STEP_APPROVALS", "FEATURE_BUNDLE_1"],
            )
        )

        assert [isinstance(s, StepTemplate) and s.is_approval_only for s in job_template.steps] == [
            True,
            False,
            False,
            False,
        ]
