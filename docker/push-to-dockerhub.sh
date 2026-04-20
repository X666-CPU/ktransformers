#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker-utils.sh"

################################################################################
# Default Configuration 🔥 关键修复：默认关闭微调，用 infer
################################################################################
CUDA_VERSION="12.8.1"
UBUNTU_MIRROR="0"
HTTP_PROXY=""
HTTPS_PROXY=""
CPU_VARIANT="x86-intel-multi"
FUNCTIONALITY="infer"  # 🔥 修复点：已经改成 infer，不再下载无效whl

DOCKERFILE="$SCRIPT_DIR/Dockerfile"
CONTEXT_DIR="$SCRIPT_DIR"
REGISTRY="docker.io"
REPOSITORY=""

DRY_RUN=false
SKIP_BUILD=false
ALSO_PUSH_SIMPLIFIED=false
MAX_RETRIES=3
RETRY_DELAY=5
EXTRA_BUILD_ARGS=()

################################################################################
# Help
################################################################################
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]
Build and push Docker image to DockerHub with standardized naming.

OPTIONS:
    --cuda-version VERSION      CUDA version (default: 12.8.1)
    --ubuntu-mirror 0|1         Use Tsinghua mirror
    --http-proxy URL            HTTP proxy
    --https-proxy URL           HTTPS proxy
    --cpu-variant VARIANT       (default: x86-intel-multi)
    --functionality TYPE        infer (推理) 或 sft (微调)
    --repository REPO           镜像仓库（必填）
    --also-push-simplified      推送简洁标签
    --dry-run                   测试运行
    -h, --help                  帮助
EOF
    exit 0
}

################################################################################
# Parse args
################################################################################
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
            --ubuntu-mirror) UBUNTU_MIRROR="$2"; shift 2 ;;
            --http-proxy) HTTP_PROXY="$2"; shift 2 ;;
            --https-proxy) HTTPS_PROXY="$2"; shift 2 ;;
            --cpu-variant) CPU_VARIANT="$2"; shift 2 ;;
            --functionality) FUNCTIONALITY="$2"; shift 2 ;;
            --dockerfile) DOCKERFILE="$2"; shift 2 ;;
            --context-dir) CONTEXT_DIR="$2"; shift 2 ;;
            --registry) REGISTRY="$2"; shift 2 ;;
            --repository) REPOSITORY="$2"; shift 2 ;;
            --skip-build) SKIP_BUILD=true; shift ;;
            --also-push-simplified) ALSO_PUSH_SIMPLIFIED=true; shift ;;
            --max-retries) MAX_RETRIES="$2"; shift 2 ;;
            --retry-delay) RETRY_DELAY="$2"; shift 2 ;;
            --dry-run) DRY_RUN=true; shift ;;
            --build-arg) EXTRA_BUILD_ARGS+=("--build-arg" "$2"); shift 2 ;;
            -h|--help) usage ;;
            *) log_error "Unknown option: $1"; exit 1 ;;
        esac
    done
}

################################################################################
# Validate
################################################################################
validate_config() {
    log_step "Validating configuration"
    check_docker_running || exit 1
    check_docker_login "$REGISTRY" || exit 1
    validate_cuda_version "$CUDA_VERSION" || exit 1

    if [ -z "$REPOSITORY" ]; then
        log_error "Repository name is required"
        exit 1
    fi

    if [ ! -f "$DOCKERFILE" ]; then
        log_error "Dockerfile not found: $DOCKERFILE"
        exit 1
    fi

    if [[ "$FUNCTIONALITY" != "sft" && "$FUNCTIONALITY" != "infer" ]]; then
        log_error "Must be 'sft' or 'infer'"
        exit 1
    fi

    log_success "Configuration validated"
}

################################################################################
# Build
################################################################################
build_image() {
    local temp_tag="ktransformers:temp-push-$(get_beijing_timestamp)"

    if [ "$SKIP_BUILD" = true ]; then
        local existing_image
        existing_image=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "ktransformers:temp-" | head -1 || echo "")
        if [ -n "$existing_image" ]; then
            echo "$existing_image"; return 0
        fi
    fi

    log_step "Building Docker image"
    local build_args=(
        --build-arg "CUDA_VERSION=$CUDA_VERSION"
        --build-arg "UBUNTU_MIRROR=$UBUNTU_MIRROR"
        --build-arg "CPU_VARIANT=$CPU_VARIANT"
        --build-arg "BUILD_ALL_CPU_VARIANTS=1"
        --build-arg "FUNCTIONALITY=$FUNCTIONALITY"
    )

    [ -n "$HTTP_PROXY" ] && build_args+=(--build-arg "HTTP_PROXY=$HTTP_PROXY")
    [ -n "$HTTPS_PROXY" ] && build_args+=(--build-arg "HTTPS_PROXY=$HTTPS_PROXY")
    build_args+=("${EXTRA_BUILD_ARGS[@]}")
    build_args+=(--network host)

    local build_cmd=(
        docker build -f "$DOCKERFILE" "${build_args[@]}" -t "$temp_tag" "$CONTEXT_DIR"
    )

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Skip build"
        return 0
    fi

    log_info "Building..."
    if "${build_cmd[@]}"; then
        log_success "Build success"
        echo "$temp_tag"
    else
        log_error "Build failed"
        exit 1
    fi
}

################################################################################
# Tags
################################################################################
generate_tags() {
    local image_tag="$1"
    local timestamp="$2"

    if [ "$DRY_RUN" = true ]; then
        local versions="SGLANG_VERSION=0.5.6
KTRANSFORMERS_VERSION=0.5.3
LLAMAFACTORY_VERSION=0.9.3"
    else
        local versions
        versions=$(extract_versions_from_image "$image_tag")
        validate_versions "$versions" || exit 1
    fi

    local full_tag
    full_tag=$(generate_image_name "$versions" "$CUDA_VERSION" "$CPU_VARIANT" "$FUNCTIONALITY" "$timestamp")
    echo "FULL_TAG=$full_tag"

    if [ "$ALSO_PUSH_SIMPLIFIED" = true ]; then
        local ktrans_ver
        ktrans_ver=$(echo "$versions" | grep "^KTRANSFORMERS_VERSION=" | cut -d= -f2)
        local simplified_tag
        simplified_tag=$(generate_simplified_tag "$ktrans_ver" "$CUDA_VERSION")
        echo "SIMPLIFIED_TAG=$simplified_tag"
    fi
}

################################################################################
# Push
################################################################################
push_image_with_retry() {
    local source_tag="$1"
    local target_tag="$2"
    local attempt=1

    log_step "Pushing $target_tag"
    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Skip push"
        return 0
    fi

    docker tag "$source_tag" "$target_tag"

    while [ $attempt -le "$MAX_RETRIES" ]; do
        log_info "Attempt $attempt/$MAX_RETRIES"
        if docker push "$target_tag"; then
            log_success "Pushed $target_tag"
            return 0
        fi
        ((attempt++))
        sleep $RETRY_DELAY
    done

    log_error "Push failed after $MAX_RETRIES attempts"
    return 1
}

################################################################################
# Main
################################################################################
main() {
    parse_args "$@"
    validate_config

    TIMESTAMP=$(get_beijing_timestamp)
    TEMP_TAG=$(build_image)
    [ "$DRY_RUN" = true ] && TEMP_TAG="ktransformers:temp-dryrun"

    TAG_INFO=$(generate_tags "$TEMP_TAG" "$TIMESTAMP")
    FULL_TAG=$(echo "$TAG_INFO" | grep "^FULL_TAG=" | cut -d= -f2)
    SIMPLIFIED_TAG=$(echo "$TAG_INFO" | grep "^SIMPLIFIED_TAG=" | cut -d= -f2 || echo "")

    FULL_IMAGE="$REGISTRY/$REPOSITORY:$FULL_TAG"
    push_image_with_retry "$TEMP_TAG" "$FULL_IMAGE"

    if [ -n "$SIMPLIFIED_TAG" ]; then
        SIMPLIFIED_IMAGE="$REGISTRY/$REPOSITORY:$SIMPLIFIED_TAG"
        push_image_with_retry "$TEMP_TAG" "$SIMPLIFIED_IMAGE"
    fi

    [ "$DRY_RUN" = false ] && cleanup_temp_images "$TEMP_TAG"

    log_success "All done!"
}

main "$@"
